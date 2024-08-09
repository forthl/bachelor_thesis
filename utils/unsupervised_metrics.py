"""Implements the training and validation steps."""
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from openTSNE import TSNE

import wandb
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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


def unsupervised_segmentation(
        segment_features: list[Tensor],
        predictions: list[Tensor],
        targets: list[Tensor],
        num_predictions: int,
        num_classes: int,
        ignore_index: int = 255,
):
    """

    Args:
        segment_features: A list of tensors of shape
            (num_segments_i, num_features) for image i
        predictions: A list of tensors of shape (H_i, W_i)(this is the original
            image size) containing the index in [0, num_segments_i - 1]
        targets: A list of tensors of shape (H_i, W_i)(this is the original
            image size) containing the index in [0, num_classes - 1] or
            ignore_index
        ignore_index: The target index to ignore when computing the loss

    Returns:

    """
    assert len(segment_features) == len(predictions) == len(targets)

    # Step 1: Cluster the segment_features
    # cluster segment_features to assign the same class to segments from
    # different images (currently each segment_feature is a different class) but
    # we want to get segment ids that correspond to object classes

    # There could be multiple segments of the same class in each image therefore
    # for the clustering it is not important to know which segment is from which
    # image, we can just flatten the segment_features

    segment_features_flat = torch.cat(segment_features, dim=0)

    # Convert to numpy array for sklearn KMeans
    segment_features_np = segment_features_flat.cpu().numpy()

    # Step 2: Cluster the segment_features using KMeans
    kmeans = KMeans(n_clusters=num_predictions, random_state=0)
    labels_np = kmeans.fit_predict(segment_features_np)

    # Convert labels back to a torch tensor
    labels = torch.tensor(labels_np, dtype=torch.long, device=segment_features_flat.device)

    # cluster the segment_features
    # TODO: Here, I would use KMeans with
    #  num_predictions clusters from sklearn
   # labels = cluster(segment_features_flat, num_predictions)

    # split the labels back to the original images
    labels_split = torch.split(labels, [t.shape[0] for t in segment_features])
    # labels: A list of tensors of shape (num_segments_i,) for image i containing
    #   the predicted class for each segment

    # use the labels to map to the predicted (clustered) classes instead of the
    # segment ids
    mapped_predictions = [
        label[prediction.long()]
        for label, prediction in zip(labels_split, predictions)
    ]
    # mapped_predictions: A list of tensors of shape (H_i, W_i) containing the
    #   predicted class for each pixel in the image

    # Step 2: Compute the metric for each possible assignment
    #   predicted class -> target class
    stats = torch.zeros(num_predictions, num_classes, dtype=torch.long)

    for prediction, target in zip(mapped_predictions, targets):
        # we remove all pixels corresponding to the ignore_index
        valid = target != ignore_index
        target_valid = target[valid]
        prediction_valid = prediction[valid]
        # This is the matrix computing the true positive for each
        #   predicted class -> target class combination
        # stat.sum(0) is the number of pixels with this target class
        # stat.sum(1) is the number of pixels with this predicted class
        # Therefore, this is enough to compute the confusion matrix
        # It currently only considers semantic segmentation (different instances
        #   of the same class are not considered as different classes)
        # TODO: Maybe we could change this to compute the optimal assignment for
	      #   instance segmentation (however, this is just to match the semantics)
        device = target_valid.device
        target_valid = target_valid.to(device)
        prediction_valid = prediction_valid.to(device)
        stat = torch.bincount(
            num_predictions * target_valid + prediction_valid,
            minlength=num_predictions * num_classes) \
            .reshape(num_classes, num_predictions).T
        stat = stat.to(device)
        stats += stat

    # Step 3: Compute the optimal assignment
    (assignments, _, _), _ = hungarian_matching(stats)

    # Step 4: Transform the predictions and compute the final instance
    #   segmentation metrics

 # E.g. https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html


    mapped_predictions = [
        prediction.cpu()
        for prediction in mapped_predictions
    ]

    predictions = [
        map_clusters(prediction.long(), assignments, num_predictions, num_classes)
        for prediction in mapped_predictions
    ]

    # TODO: predictions need to be transformed to the torchmetrics format,
    #  in particular for instance segmentation (for this you need the original
    #  predictions)
    # update the instance segmentation metrics from torchmetrics
    # See https://cocodataset.org/#detection-eval and the corresponding paper
    # on how to evaluate instance segmentation

    return predictions, assignments
