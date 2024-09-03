import numpy as np

import open3d as o3d
import torch

from utils import cluster_algorithms
from utils.utils_folder.stego_utils import filter_big_classes


# get masks from segmentations from gep_seg
def get_segmentation_masks_geo_seg(geo_seg):
    masks = []
    for c in np.unique(geo_seg):
        segmentation = geo_seg == c
        masks.append(segmentation)
    return masks


class_colors = {
    (0, 0, 0): 0,  # unlabeled
    (220, 20, 60): 17,  # person 1
    (255, 0, 0): 18,  # rider 1
    (0, 0, 142): 19,  # car 1
    (0, 0, 70): 20,  # truck 1
    (0, 60, 100): 21,  # bus 1
    #(0, 0, 90): 22,  # caravan
    #(0, 0, 110): 23,  # trailer
    (0, 80, 100): 24,  # train 1
    (0, 0, 230): 25,  # motorcycle 1
    (119, 11, 32): 26  # bicycle 1
}


# get masks from segmentations
def get_segmentation_masks(img):
    masks = []
    classes = []
    img_rgb = np.asarray(img)

    I = img.convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        masks.append(segmentation)

        rgb_color = tuple(np.mean(img_rgb[segmentation], axis=0).astype(int))
        classes.append(class_colors[rgb_color])
        segmentation_size = img_rgb[segmentation].shape

    return masks, classes


# mask depth image with segmentations
def get_masked_depth(depth_map, masks):
    masked_depths = []

    for mask in masks:
        seg_masked = np.where(mask, depth_map, 0)
        masked_depths.append(seg_masked)
    return masked_depths


def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array([non_zero[0], non_zero[1], mask[non_zero[0], non_zero[1]]])
        point_cloud = np.transpose(point_cloud)
        point_clouds.append(point_cloud)
    return point_clouds


def create_all_point_clouds(depth):
    non_zero = np.nonzero(depth)
    point_cloud = np.array([non_zero[0], non_zero[1], depth[non_zero[0], non_zero[1]]])
    point_cloud = np.transpose(point_cloud)
    return point_cloud


def create_projected_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        point_cloud = project_disparity_to_3d(mask)
        point_clouds.append(point_cloud)
    return point_clouds


def unproject_point_cloud(data):
    focal_length_x = 2262.52 / 3.2
    focal_length_y = 2265.3017905988554 / 3.2
    cx = 1096.98 / 6.4
    cy = 513.137 / 3.2

    for point in data:
        point[0] = int(round((point[0] * focal_length_x / point[2]) + cx))
        point[1] = int(round((point[1] * focal_length_y / point[2]) + cy))

    data = data[:, [1, 0, 2]]  # convert from xyz coordinates to array indexes
    return data.astype('int')


def project_disparity_to_3d(depth_map):  # debug this shit cause the rescaling is wrong
    focal_length_x = 2262.52 / 3.2
    focal_length_y = 2265.3017905988554 / 3.2
    cx = 1096.98 / 6.4
    cy = 513.137 / 3.2

    height, width = depth_map.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Filter out points with disparity value of 0
    valid_indices = np.where(depth_map != 0)
    depth = depth_map[valid_indices] * 1000 / 0.2645833333
    depth = depth / 3.2
    points_x = (grid_x[valid_indices] - cx) * (depth / focal_length_x)
    points_y = (grid_y[valid_indices] - cy) * (depth / focal_length_y)
    points_z = depth

    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)

    return point_cloud




def segmentation_to_instance_mask(filtered_segmentation_mask, depth_map, image_shape, clustering_algorithm, epsilon,
                                  min_samples, max_eps, metric, cluster_method, n_clusters,max_k, bgmm_weights_threshold, covariance_type, init_params, project_data=False, filtering_big_classes = False):
    labels_list = []

    class_masks, classes = get_segmentation_masks(filtered_segmentation_mask)
    class_masks.pop(0) # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road, building)
    classes.pop(0) # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road, building)

    if filtering_big_classes:
        class_masks, classes = filter_big_classes(class_masks, classes)

    masked_depths = get_masked_depth(depth_map, class_masks)

    point_clouds = []

    if project_data:
        point_clouds = create_projected_point_clouds(masked_depths)
    else:
        point_clouds = create_point_clouds(masked_depths)

    instance_mask = np.zeros(image_shape)
    current_num_instances = 0



    for Idx, point_cloud in enumerate(point_clouds):

        if point_cloud.shape[0] <= 1:  # TODO check if it is an empty point cloud. Look into this bug later
            continue

        max_k = min(max_k, point_cloud.shape[0])

        if clustering_algorithm == "bgmm":
            cl = cluster_algorithms.BayesianGaussianMixtureModel(data=point_cloud, max_k=max_k,  bgmm_weights_threshold=bgmm_weights_threshold, covariance_type=covariance_type, init_params=init_params)
        elif clustering_algorithm == "dbscan":
            cl = cluster_algorithms.Dbscan(point_cloud, epsilon, min_samples)
        elif clustering_algorithm == "optics":
            cl = cluster_algorithms.Optics(point_cloud, min_samples=min(min_samples,point_cloud.shape[0]), max_eps=max_eps, metric=metric, cluster_method=cluster_method)
        elif clustering_algorithm == "kmeans":
            cl = cluster_algorithms.Kmeans(point_cloud, n_clusters=n_clusters)

        try:
            labels = cl.find_clusters()
        except:
            labels = np.zeros(point_cloud.shape[0])

        labels += 1

        if project_data:
            point_cloud = unproject_point_cloud(
                point_cloud)

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
    meyer = torch.unique(torch.tensor(instance_mask))

    return instance_mask, labels_list, instance_list


def remove_point_cloud_outliers(point_cloud):
    # removes all points that don't have less than np_points in their neighborhood of radius
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Radius outlier removal:
    pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=50, radius=1)
    outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
    outlier_rad_pcd.paint_uniform_color([1., 0., 1.])
    pcdnp = np.asarray(pcd_rad.points)

    return pcdnp, ind_rad