import numpy as np

import open3d as o3d

from utils import cluster_algorithms_new


# get masks from segmentations from gep_seg
def get_segmentation_masks_geo_seg(geo_seg):
    masks = []
    for c in np.unique(geo_seg):
        segmentation = geo_seg == c
        masks.append(segmentation)
    return masks


# get masks from segmentations
def get_segmentation_masks(img):
    masks = []
    I = img.convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        masks.append(segmentation)
    return masks


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

    data = data[:,[1,0,2]]# convert from xyz coordinates to array indexes
    return data.astype('int')


def project_disparity_to_3d(depth_map):#debug this shit cause the rescaling is wrong
    focal_length_x = 2262.52/3.2
    focal_length_y = 2265.3017905988554/3.2
    cx = 1096.98/6.4
    cy = 513.137/3.2


    height, width = depth_map.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Filter out points with disparity value of 0
    valid_indices = np.where(depth_map != 0)
    depth = depth_map[valid_indices]*1000/0.2645833333
    depth = depth/3.2
    points_x = (grid_x[valid_indices] - cx) * (depth / focal_length_x)
    points_y = (grid_y[valid_indices] - cy) * (depth / focal_length_y)
    points_z = depth

    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)

    return point_cloud


def segmentation_to_instance_mask(filtered_segmentation_mask, depth_map, image_shape, clustering_algorithm, epsilon ,min_samples ,project_data=False):

    class_masks = get_segmentation_masks(filtered_segmentation_mask)
    class_masks.pop(0)  # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road, building)
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

        max_k = min(20, point_cloud.shape[0])

        if clustering_algorithm == "bgmm":
            cl = cluster_algorithms_new.BayesianGaussianMixtureModel(data=point_cloud, max_k=max_k)
        elif clustering_algorithm == "dbscan":
            cl = cluster_algorithms_new.Dbscan(point_cloud, epsilon, min_samples)

        try:
           labels = cl.find_clusters()
        except:
            labels = np.zeros(point_cloud.shape[0])

        labels += 1

        if project_data:
            point_cloud = unproject_point_cloud(
                point_cloud)  # maybe try just projecting the data and getting the labels and keeping an unprojected copy of the data

        class_instance_mask = np.zeros(image_shape)

        for index, point in enumerate(point_cloud):
            class_instance_mask[int(point[0]), int(point[1])] = labels[index]

        num_clusters = len(set(labels)) - 1
        class_instance_mask = np.where(class_instance_mask != 0, class_instance_mask + current_num_instances, 0)
        current_num_instances += num_clusters
        instance_mask = np.add(instance_mask, class_instance_mask)

    return instance_mask


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