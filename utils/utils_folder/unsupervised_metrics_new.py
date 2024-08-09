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
from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import utils.evaluation_utils as eval_utils



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

    def __init__(self, num_classes, num_predictions=None, ignore_index=255, mode='cosine'):
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
        self.segment_features.append(segment_features)
        self.predictions.append(predictions)
        self.targets.append(targets)


    def compute(self):

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
        assert len(self.segment_features) == len(self.predictions) == len(self.targets)

        # Step 1: Cluster the segment_features
        # cluster segment_features to assign the same class to segments from
        # different images (currently each segment_feature is a different class) but
        # we want to get segment ids that correspond to object classes

        # There could be multiple segments of the same class in each image therefore
        # for the clustering it is not important to know which segment is from which
        # image, we can just flatten the segment_features

        segment_features_flat = torch.cat(self.segment_features, dim=0)

        # Convert to numpy array for sklearn KMeans
        segment_features_np = segment_features_flat.cpu().numpy()

        # Step 2: Cluster the segment_features using KMeans
        kmeans = KMeans(n_clusters=self.num_predictions, random_state=0)
        labels_np = kmeans.fit_predict(segment_features_np)

        # Convert labels back to a torch tensor
        labels = torch.tensor(labels_np, dtype=torch.long, device=segment_features_flat.device)

        # cluster the segment_features
        # TODO: Here, I would use KMeans with
        #  num_predictions clusters from sklearn
       # labels = cluster(segment_features_flat, num_predictions)

        # split the labels back to the original images
        labels_split = torch.split(labels, [t.shape[0] for t in self.segment_features])
        # labels: A list of tensors of shape (num_segments_i,) for image i containing
        #   the predicted class for each segment

        # use the labels to map to the predicted (clustered) classes instead of the
        # segment ids
        mapped_predictions = [
            label[prediction.long()]
            for label, prediction in zip(labels_split, self.predictions)
        ]
        # mapped_predictions: A list of tensors of shape (H_i, W_i) containing the
        #   predicted class for each pixel in the image
        image_wise_metrics = []

        # Step 2: Compute the metric for each possible assignment
        #   predicted class -> target class
        stats = torch.zeros(self.num_predictions, self.num_classes, dtype=torch.long)

        image_assignments = []

        for prediction, target in zip(mapped_predictions, self.targets):
            # we remove all pixels corresponding to the ignore_index
            valid = target != self.ignore_index
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
                self.num_predictions * target_valid + prediction_valid,
                minlength=self.num_predictions * self.num_classes) \
                .reshape(self.num_classes, self.num_predictions).T
            stat = stat.to(device)
            stats += stat
            (image_assignment, _, _), metrics = hungarian_matching(stat)
            image_assignments.append(image_assignment)
            metrics['Adjusted Rand Index'] = adjusted_rand_index(stat).item()
            image_wise_metrics.append(metrics)

        # Step 3: Compute the optimal assignment
        (assignments, histogram, _), metric_dict = hungarian_matching(stats)

        # Step 4: Transform the predictions and compute the final instance
        #   segmentation metrics

     # E.g. https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html




        mapped_predictions = [
            prediction.cpu()
            for prediction in mapped_predictions
        ]

        predictions = [
            map_clusters(prediction.long(), assignments, self.num_predictions, self.num_classes)
            for prediction in mapped_predictions
        ]


        return predictions, self.targets


