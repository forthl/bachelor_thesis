"""Implements the training and validation steps."""
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from openTSNE import TSNE

import wandb
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from test_time_augmentations import TestTimeAugmentation
from tools import get_attention

def cluster(features, num_predictions):
    # TODO: rename variables
    print('Clustering...')
    best_log_likelihood = -np.inf
    best_labels = None
    for _ in range(1):
        clustering = GaussianMixture(n_components=2)
        label2 = torch.as_tensor(clustering.fit_predict(features))
        means = torch.as_tensor(clustering.means_)
        covs = torch.as_tensor(clustering.covariances_)
        multivariate_normal = MultivariateNormal(means, covs)
        probs = multivariate_normal.log_prob(features[:, None, :])
        i = probs[label2 == 0, 0].mean() > probs[label2 == 1, 1].mean()
        features2 = features[label2 == i]
        clustering2 = GaussianMixture(n_components=num_predictions - 1)
        label3 = torch.as_tensor(clustering2.fit_predict(features2))
        label_out = torch.zeros_like(label2, dtype=torch.long)
        label_out[label2 == i] = label3 + 1
        weights_full = torch.as_tensor(
            [clustering.weights_[i.bitwise_not()]] + (clustering.weights_[i] * clustering2.weights_).tolist())
        covs_full = torch.as_tensor(
            np.concatenate([clustering.covariances_[i.bitwise_not()][None], clustering2.covariances_]))
        means_full = torch.as_tensor(np.concatenate([clustering.means_[i.bitwise_not()][None], clustering2.means_]))
        mixture = MixtureSameFamily(Categorical(weights_full), MultivariateNormal(means_full, covs_full))
        log_likelihood = mixture.log_prob(features).mean()
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_labels = label_out
    print('Done clustering.')
    return best_labels, best_log_likelihood

def adjusted_rand_index(
        n: torch.Tensor,
):
    """Computes the adjusted rand index for a batch of predictions."""

    def n_choose_2(n):
        return n * (n - 1) / 2

    a = n.sum(dim=-1)
    b = n.sum(dim=-2)

    a_choose_2_sum = n_choose_2(a).sum(dim=-1)
    b_choose_2_sum = n_choose_2(b).sum(dim=-1)
    abn = a_choose_2_sum * b_choose_2_sum / n_choose_2(n.sum(dim=(-1, -2)))
    out = (n_choose_2(n).sum(dim=(-1, -2)) - abn) / (0.5 * (a_choose_2_sum + b_choose_2_sum) - abn)
    return out.nan_to_num(1.)

def hungarian_matching(stats, normalized=True):
    num_predictions, num_classes = stats.shape
    if normalized:
        eps = torch.finfo().eps
        normalized_stats = stats / (stats.sum(dim=0, keepdims=True) + stats.sum(dim=1, keepdims=True) - stats + eps)
    else:
        normalized_stats = stats
    assignments = linear_sum_assignment(normalized_stats, maximize=True)
    if num_predictions == num_classes:
        histogram = stats[np.argsort(assignments[1]), :]
        assignments_t = None
    elif num_predictions > num_classes:
        assignments_t = linear_sum_assignment(normalized_stats.T, maximize=True)
        histogram = stats[assignments_t[1], :]
        missing = list(set(range(num_predictions)) - set(assignments[0]))
        new_row = stats[missing, :].sum(0, keepdim=True)
        histogram = torch.cat([histogram, new_row], dim=0)
        new_col = torch.zeros(num_classes + 1, 1, device=histogram.device)
        histogram = torch.cat([histogram, new_col], dim=1)

    tp = torch.diag(histogram)
    fp = torch.sum(histogram, dim=0) - tp
    fn = torch.sum(histogram, dim=1) - tp

    iou = tp / (tp + fp + fn)
    # torch.set_printoptions(sci_mode=False)
    # print(iou)
    opc = torch.sum(tp) / torch.sum(histogram)

    metric_dict = {"mIoU": iou[~torch.isnan(iou)].mean().item(),
                   "Accuracy": opc.item()}
    return (assignments, histogram, assignments_t), metric_dict

def map_clusters(clusters, assignments, num_predictions, num_classes):
    if num_predictions == num_classes:
        return torch.tensor(assignments[1])[clusters]
    else:
        missing = sorted(list(set(range(num_predictions)) - set(assignments[0])))
        cluster_to_class = assignments[1]
        for missing_entry in missing:
            if missing_entry == cluster_to_class.shape[0]:
                cluster_to_class = np.append(cluster_to_class, -1)
            else:
                cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
        cluster_to_class = torch.tensor(cluster_to_class)
        return cluster_to_class[clusters]

def reduce_dict(dicts, function=torch.mean, prefix=''):
    out = {}
    for key in dicts[0].keys():
        out[prefix + key] = function(torch.as_tensor([d[key] for d in dicts]))
    return out

class UnsupervisedMetrics(Metric):
    is_differentiable: Optional[bool] = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, num_classes, num_predictions=None, ignore_index=21, mode='cosine'):
        super().__init__(dist_sync_on_step=True)
        self.num_classes = num_classes
        self.num_predictions = num_predictions if num_predictions is not None else num_classes
        self.ignore_index = ignore_index
        self.mode = mode

        self.add_state("segment_features", default=[], dist_reduce_fx=None)
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, segment_features, predictions, targets):
        self.segment_features.extend(segment_features)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    @torch.no_grad()
    def compute(self):
        segment_features_flat = torch.cat(self.segment_features)
        labels, log_likelihood = cluster(segment_features_flat, self.num_predictions)
        # labels = torch.as_tensor(GaussianMixture(n_components=self.num_predictions, random_state=0).fit_predict(segment_features_flat))
        # labels = torch.as_tensor(KMeans(n_clusters=self.num_predictions, random_state=0).fit_predict(segment_features_flat))
        labels_split = torch.split(labels, [t.shape[0] for t in self.segment_features])
        mapped_predictions = [
            label[prediction.long()]
            for label, prediction in zip(labels_split, self.predictions)
        ]

        print('Computing metrics...')
        image_wise_metrics = []
        stats = torch.zeros(self.num_predictions, self.num_classes, dtype=torch.long)
        image_assignments = []
        for prediction, target in zip(mapped_predictions, self.targets):
            valid = (target != self.ignore_index)
            target_valid = target[valid]
            prediction_valid = prediction[valid]
            stat = torch.bincount(
                self.num_predictions * target_valid + prediction_valid,
                minlength=self.num_predictions * self.num_classes) \
                .reshape(self.num_classes, self.num_predictions).T
            stats += stat
            (image_assignment, _, _), metrics = hungarian_matching(stat)
            image_assignments.append(image_assignment)
            metrics['Adjusted Rand Index'] = adjusted_rand_index(stat).item()
            image_wise_metrics.append(metrics)

        (self.assignments, histogram, _), metric_dict = hungarian_matching(stats)
        print('Done computing metrics.')

        embeddings = []
        for segment_feature, prediction, label, target, image_assignment in \
                zip(self.segment_features, self.predictions, labels_split,
                    self.targets, image_assignments):
            image_embdeddings = []
            for index, feature in enumerate(segment_feature):
                binary_prediction = prediction == index
                image = torch.ones_like(prediction) * (self.num_classes + 1)
                class_index = map_clusters(label[index], self.assignments, self.num_predictions, self.num_classes)
                image[binary_prediction] = class_index
                image_matching = map_clusters(label[index], image_assignment, self.num_predictions, self.num_classes)
                image_embdeddings.append([image, int(image_matching.item()), int(class_index.item())])
            embeddings.append(image_embdeddings)

        metric_dict = {
            # 'log_likelihood': log_likelihood.item() / 100,
            **reduce_dict(image_wise_metrics, function=torch.mean, prefix='average '),
            **metric_dict
        }
        image_wise_metrics = reduce_dict(image_wise_metrics, function=lambda x: x, prefix='image ')
        self.predictions = [map_clusters(prediction.long(), self.assignments, self.num_predictions, self.num_classes)
                            for prediction in mapped_predictions]
        return {k: 100 * v for k, v in metric_dict.items()}, image_wise_metrics, embeddings

class UnsupervisedSegmentation(pl.LightningModule):
    """Implements the training and validation steps for unsupervised segmentation."""

    def __init__(self,
                 pretrained_feature_model,
                 model,
                 optimizer,
                 lr_scheduler,
                 augmentation=None,
                 max_num_clusters=10,
                 num_classes=22,
                 ignore_index=21):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_feature_model', 'model', 'lr_scheduler'])

        self.pretrained_feature_model = pretrained_feature_model.requires_grad_(False)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if augmentation is None:
            augmentation = TestTimeAugmentation()
        self.augmentation = augmentation

        self.max_num_clusters = max_num_clusters
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.metrics = UnsupervisedMetrics(21, 21)

        self.num_full_resolution_images = 8
        self.downsampling_scale_factor = 0.25

    def training_step(self, batch, batch_idx):
        """Performs a training step.

        First multiple augmentations are computed and the features are aggregated. Next, K-Means is applied and the
        centroids and labels are used to compute the loss. The loss is then logged to wandb.
        """
        images = batch[0]
        B, _, H, W = images.shape
        self.pretrained_feature_model.eval()
        with torch.no_grad():
            # batch: B x 3 x H x W
            transformed_images = self.augmentation.transform_images(images)
            pretrained_features = get_attention(transformed_images, self.pretrained_feature_model, -1)
            pretrained_features = self.augmentation.transform_features(pretrained_features)
            _, C, h, w = pretrained_features.shape
            pretrained_features = pretrained_features / pretrained_features.norm(dim=1, keepdim=True)

            pretrained_features_flat = pretrained_features.view(B, C, h * w).permute(0, 2, 1)
            # pretrained_features: B x hw x C

            num_clusters = np.random.randint(2, self.max_num_clusters + 1)
            mode = np.random.choice(['euclidean', 'cosine'])
            kmeans = KMeans(n_clusters=num_clusters, mode=mode)
            labels = kmeans.fit_predict(pretrained_features_flat)

            hw = pretrained_features_flat.shape[-2]
            fraction = np.sqrt((H * W) / hw)
            h = int(H / fraction)
            w = int(W / fraction)
            labels = labels.view(-1, h, w)

            centroids = kmeans.centroids
            centroids = centroids / centroids.norm(dim=-1, keepdim=True)
            # centroids: B x num_clusters x C

        loss = self.model.compute_loss(images, centroids, labels, pretrained_features)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
                "strict": False,
                "name": None
            }
        }

    def validation_step(self, batch, batch_idx):
        """Performs a validation step."""
        images, targets = batch
        # predicted, confidences = self.model(images)
        target_classes = [set(t.view(-1).cpu().numpy()) for t in targets]
        for c in target_classes:
            c.remove(self.ignore_index)
        num_clusters = [len(c) for c in target_classes]
        probabilities, features = self.model(images, num_clusters)
        # probabilities: list of length B with tensors of shape num_features x H x W
        # features: list of length B with tensors of shape num_features x C
        features = [feature.cpu() for feature in features]
        targets = [target[0].cpu() for target in targets]
        probabilities = [F.interpolate(p.float()[None].cpu(), size=target.shape, mode='bicubic')[0]
                         for target, p in zip(targets, probabilities)]

        predictions = [p.argmax(dim=0) for p in probabilities]

        self.metrics.update(features, predictions, targets)

        images_out = []
        confidences_out = []
        for i, (image, p) in enumerate(zip(images, probabilities)):
            confidence = p.max(dim=0).values
            image = image.cpu()
            if i + batch_idx * len(images) >= self.num_full_resolution_images:
                image = F.interpolate(image[None], scale_factor=self.downsampling_scale_factor, mode='bicubic')[0]
                confidence = F.interpolate(confidence[None][None], scale_factor=self.downsampling_scale_factor, mode='bicubic')[0][0]
            images_out.append(image)
            confidences_out.append(confidence)
        return {'images': images_out, 'confidences': confidences_out}

    def validation_epoch_end(self, outputs):
        metrics, image_wise_metrics, embeddings = self.metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True)

        for key, values in image_wise_metrics.items():
            data = [(i, value) for i, value in enumerate(values)]
            table = wandb.Table(data=data, columns=['image id', key])
            wandb.log({key: wandb.plot.line(table, 'image id', key)})

        images = [image for output in outputs for image in output['images']]
        confidences = [confidence for output in outputs for confidence in output['confidences']]

        table = wandb.Table(columns=['image', 'confidence', 'prediction', 'ground truth'])
        embeddings_table = wandb.Table(columns=['segmentation', 'image-wise matching', 'prediction', 'embedding'])

        class_labels = self.trainer.datamodule.class_labels
        extended_class_labels = {self.num_classes + 1: 'other', **class_labels}

        print('Computing embeddings...')
        all_features = torch.cat(self.metrics.segment_features, dim=0)
        embedding = TSNE(perplexity=10., n_iter=10_000, n_jobs=-1, verbose=4).fit(all_features)
        embeddings_split = torch.split(
            torch.as_tensor(embedding),
            [t.shape[0] for t in self.metrics.segment_features]
        )
        print('Done computing embeddings.')

        print('Logging media...')
        for i, (image, confidence, prediction, target, embedding, segment_features) in enumerate(
                zip(images, confidences, self.metrics.predictions, self.metrics.targets,
                    embeddings, embeddings_split)):
            unnormalized_image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device)[:, None, None] \
                                 + torch.tensor([0.485, 0.456, 0.406], device=image.device)[:, None, None]
            plot_image = (unnormalized_image * 255).round()

            wandb_image = wandb.Image(plot_image)

            confidence = confidence.cpu().numpy()[None]
            confidence = (confidence * 255).astype(np.uint8)
            confidence = wandb.Image(confidence)

            prediction = F.interpolate(prediction[None][None].float(), size=plot_image.shape[1:], mode='nearest')[0][0].long()
            prediction = wandb.Image(plot_image, masks={
                'prediction': {
                    "mask_data": prediction.cpu().numpy(),
                    "class_labels": class_labels
                }
            })

            target = F.interpolate(target[None][None].float(), size=plot_image.shape[1:], mode='nearest')[0][0].long()
            ground_truth = wandb.Image(plot_image, masks={
                'ground truth': {
                    "mask_data": target.cpu().numpy(),
                    "class_labels": class_labels
                },
            })

            table.add_data(wandb_image, confidence, prediction, ground_truth)

            for (binary_segmentation, image_wise_matching, class_index), feature in zip(embedding, segment_features):
                binary_segmentation = F.interpolate(binary_segmentation[None][None].float(), size=plot_image.shape[1:], mode='nearest')[0][0].long()
                segmentation = wandb.Image(plot_image, masks={
                    'segmentation': {
                        "mask_data": binary_segmentation.cpu().numpy(),
                        "class_labels": extended_class_labels
                    }
                })
                embeddings_table.add_data(
                    segmentation,
                    class_labels[image_wise_matching],
                    class_labels[class_index],
                    feature.tolist())

        wandb.log({'images': table, 'embeddings': embeddings_table})
        print('Done logging media.')
        self.metrics.reset()
