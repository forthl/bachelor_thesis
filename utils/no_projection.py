from utils import cluster_algorithms
from utils.maskDepth_new import get_segmentation_masks
from utils.utils_folder.stego_utils import filter_big_classes
import numpy as np
import torch
def segmentation_to_instance_mask(filtered_segmentation_mask, image_shape, clustering_algorithm, epsilon,
                                  min_samples, max_eps, metric, cluster_method, n_clusters, max_k, bgmm_weights_threshold, covariance_type, init_params,
                                  filtering_big_classes=False):
    labels_list = []

    # Erhalte die Masken und Klassen aus der segmentierten Maske
    class_masks, classes = get_segmentation_masks(filtered_segmentation_mask)
    class_masks.pop(0)  # Entferne die Maske mit Klassen ohne Attribute (z.B. Straße, Gebäude)
    classes.pop(0)  # Entferne die entsprechende Klasse ohne Attribute

    if filtering_big_classes:
        class_masks, classes = filter_big_classes(class_masks, classes)

    instance_mask = np.zeros(image_shape)
    current_num_instances = 0

    for Idx, class_mask in enumerate(class_masks):
        # Erzeuge eine Punktwolke aus der 2D-Segmentierungsmaske
        point_cloud = create_2d_point_cloud(class_mask)

        if point_cloud.shape[0] <= 1:  # Überprüfe, ob die Punktwolke leer ist
            continue

        max_k = min(max_k, point_cloud.shape[0])

        if clustering_algorithm == "bgmm":
            cl = cluster_algorithms.BayesianGaussianMixtureModel(data=point_cloud, max_k=max_k, bgmm_weights_threshold=bgmm_weights_threshold, covariance_type=covariance_type, init_params=init_params)
        elif clustering_algorithm == "dbscan":
            cl = cluster_algorithms.Dbscan(point_cloud, epsilon, min_samples)
        elif clustering_algorithm == "optics":
            cl = cluster_algorithms.Optics(point_cloud, min_samples, max_eps=max_eps, metric=metric, cluster_method=cluster_method)
        elif clustering_algorithm == "kmeans":
            cl = cluster_algorithms.Kmeans(point_cloud, n_clusters=n_clusters)

        try:
            labels = cl.find_clusters()
        except:
            labels = np.zeros(point_cloud.shape[0])

        labels += 1

        class_instance_mask = np.zeros(image_shape)

        for index, point in enumerate(point_cloud):
            class_instance_mask[int(point[0]), int(point[1])] = labels[index]

        num_clusters = len(set(labels))
        if 0 in labels:
            num_clusters -= 1
        class_instance_mask = np.where(class_instance_mask != 0, class_instance_mask + current_num_instances, 0)

        instance_mask = np.add(instance_mask, class_instance_mask)
        for i in range(current_num_instances, current_num_instances + num_clusters):
            labels_list.append(classes[Idx])
        current_num_instances += num_clusters

    instance_list = list(range(1, current_num_instances + 1))
    #unique_instances = torch.unique(torch.tensor(instance_mask))

    return instance_mask, labels_list, instance_list

def create_2d_point_cloud(mask):
    # Erzeuge eine Punktwolke aus der 2D-Segmentierungsmaske
    points = np.column_stack(np.where(mask > 0))  # Alle Pixelkoordinaten mit Werten > 0
    return points
