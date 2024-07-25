import torch

import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import cv2

from utils.utils_folder.quad_tree import Point, Node, Quad
from utils.utils_folder.pc_utils import write_ply


def generateRangeImageFromPointCloud(pointcloud, shape):
    range_image = torch.zeros(shape)

    for p in pointcloud:
        range_image[p[0]][p[1]] = p[2].float()

    return range_image
    
def generateRangeImageFromTree(tree: Quad):
    nodes = tree.gather()
    
    range_image = torch.zeros((tree.bot_right.x, tree.bot_right.y))
    
    for n in nodes:
        if n.pos is not None:
            range_image[n.pos.x][n.pos.y] = n.data 

    return range_image


def generateQuadTreeFromRangeImage(range_image, insert=False):
    tree = Quad(Point(), Point(
        range_image.shape[0] + 1, range_image.shape[1] + 1))

    if insert:
        for y in range(range_image.shape[0]):
            rows = range_image[y].nonzero()
            
            if len(rows) <= 0:
                continue
            
            for x in rows:
                tree.insert(Node(Point(y, x), range_image[y][x]))

    return tree

def generatePointCloudFromRangeImage(range_image):
    depth_instensity = np.array(256 * range_image / 0x0fff,
                            dtype=np.uint8)
    
    # iio.imwrite('outputs/grayscale.png', depth_instensity)
    
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(range_image, cmap="gray")
    # axs[0].set_title('Depth image')
    # axs[1].imshow(depth_instensity, cmap="gray")
    # axs[1].set_title('Depth grayscale image')
    # plt.show()
    
    pcd = []
    
    # height, width = range_image.shape
    height, width = range_image.shape
    for i in range(height):
       for j in range(width):
           z = depth_instensity[i][j]
           x = j
           y = i
           if z > 0:
            pcd.append([x, y, z])
    
    pc = torch.tensor(pcd).float()
    
    return pc.T

def project_disparity_to_3d(disparity_map_dir, mask):
    disparity_map = cv2.imread(disparity_map_dir, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity_map[disparity_map > 0] = np.array(256 * disparity_map[disparity_map > 0] / 0x0fff, dtype=np.uint8).astype(np.float32)
    # disparity_map[disparity_map > 0] = (disparity_map[disparity_map > 0] - 1) / 256
                 
    disparity = torch.zeros(disparity_map.shape)   

    # disparity = disparity_map
    disparity =  disparity_map * mask

        
    fig, axs = plt.subplots(1, 1)
    axs.imshow(disparity, cmap="gray")
    axs.set_title('Depth image')
    #plt.show()

    focal_length = 2262.52
    cx = 1096.98
    cy = 513.137
    baseline = 0.209313

    height, width = disparity.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the depth from disparity
    depth = disparity

    # Filter out points with disparity value of 0
    valid_indices = np.where(disparity != 0)
    depth = depth[valid_indices]
    points_x = (grid_x[valid_indices] - cx) * depth / focal_length
    points_y = (grid_y[valid_indices] - cy) * depth / focal_length
    points_z = depth


    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)
    
    plot_point_cloud(point_cloud)

    return torch.tensor(point_cloud).T

def plot_point_cloud(point_cloud):
    write_ply(point_cloud, 'pointcloud.ply', False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Plot the point cloud
    ax.scatter(x, y, z, c=z, cmap='jet', s=1)

    plt.xlim(-6, 6)
    plt.ylim(-2, 2)
    ax.set_zlim(0, 32)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    # Show the plot
    #plt.show()

def imageFromPointsAndClassification(pc, classification, range_image):    
    img = np.zeros(range_image.shape)
        
    pc = pc.int().cpu().numpy()
    
    for i in range(pc.shape[2]):
        img[pc[0, 1, i].data, pc[0, 0, i].data] = classification[i] + 1
    
    fig, axs = plt.subplots(1, 2)
    
    axs[0].imshow(img)
    axs[0].set_title('Instances')
    axs[1].imshow(range_image, cmap="gray")
    axs[1].set_title('Depth')
    #plt.show()
    
    # plot_point_cloud(pc)
    
    print(np.unique(classification) > 1) 