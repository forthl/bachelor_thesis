import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import NearestNDInterpolator
from utils.utils_folder.quad_tree import Point, Node, Quad
from utils.utils_folder.data_utils import generateQuadTreeFromRangeImage, generateRangeImageFromTree

from utils.utils_folder.representation import plotGraph

import scipy.ndimage as ndimage


def removeGround(range_image):
    tans = np.zeros(range_image.shape)
    height, width = range_image.shape

    for c in range(width):
        c_idx = height - c - 1
        col_indeces = range_image[:, c_idx].nonzero()
        col_indeces = col_indeces[0]

        if len(col_indeces) == 0:
            continue

        for i in range(col_indeces.shape[0]):
            idx = col_indeces.shape[0] - i - 1

            if i == 0:
                # tans.insert(Node(Point(col_indeces[idx].item(), c), 1.5))
                continue

            idx_A = col_indeces[idx + 1].item()
            idx_B = col_indeces[idx].item()

            z_A = range_image[idx_A][c_idx]
            z_B = range_image[idx_B][c_idx]

            epsilon_a = angle_between(
                np.array([c_idx, idx_A, -z_A]), np.array([c_idx, 1, 0]))
            epsilon_b = angle_between(
                np.array([c_idx, idx_B, -z_B]), np.array([c_idx, 1, 0]))

            delta_z = np.abs(z_A * np.sin(epsilon_a) - z_B * np.sin(epsilon_b))
            delta_x = np.abs(z_A * np.cos(epsilon_a) - z_B * np.cos(epsilon_b))

            tan = torch.atan2(torch.tensor(delta_z),
                              torch.tensor(delta_x)).item()

            tans[idx_A][c_idx] = tan

    # plotGraph(tans)

    labels = torch.zeros(range_image.shape)

    for c in range(width):
        idx = c
        col_indeces = range_image[:, idx].nonzero()
        col_indeces = col_indeces[0]

        if col_indeces.shape[0] <= 1:
            continue

        labelGround(col_indeces[-1].item(), idx, labels, tans)

    # plotGraph(labels)

    no_ground = torch.abs(labels - 1) * range_image

    return no_ground, labels


def labelGround(y, x, labels, tans):
    q = []
    p = (y, x)

    height, width = tans.shape
    distance = 1

    q.append(p)

    while len(q) > 0:
        node = q[0]

        y, x = node

        labels[node] = 1
        neighbors = np.mgrid[max(y - distance, 0):min(y + distance, height),
                    max(x - distance, 0):min(x + distance, width)]

        for p_y in np.unique(neighbors[0].flatten()):
            for p_x in np.unique(neighbors[1].flatten()):
                n_p = (p_y, p_x)
                if n_p in q:
                    continue
                if labels[n_p] == 1:
                    continue
                if tans[n_p] == 0:
                    labels[n_p] = 1
                    continue

                if np.abs(tans[node] - tans[n_p]) < 0.0022:
                    q.append(n_p)

        q = q[1:]


def labelRangeImage(range_image):
    labels = np.zeros(range_image.shape)

    l = 1

    height, width = range_image.shape

    for x in range(width):
        for y in range(height):
            n = (y, x)
            if labels[n] <= 0 and range_image[n] > 0:
                labelSegments(n, range_image, labels, l)
                l += 1

    # plotGraph(labels)

    return labels


def labelSegments(n: Node, range_image, labels, label):
    q = [n]
    height, width = range_image.shape
    distance = 3

    while len(q) > 0:
        n = q[0]
        y, x = n

        labels[n] = label

        neighbors = np.mgrid[max(y - distance, 0):min(y + distance, height),
                    max(x - distance, 0):min(x + distance, width)]

        for p_y in np.unique(neighbors[0].flatten()):
            for p_x in np.unique(neighbors[1].flatten()):
                nn = (p_y, p_x)

                if nn in q:
                    continue

                if labels[nn] > 0:
                    continue

                if range_image[nn] <= 0:
                    continue

                d1 = torch.tensor(max(range_image[n], range_image[nn]))
                d2 = torch.tensor(min(range_image[n], range_image[nn]))

                phi = angle_between(np.array([x, y, range_image[n]]),
                                    np.array([p_x, p_y, range_image[nn]])) if range_image[n] > range_image[
                    nn] else angle_between(np.array(
                    [p_x, p_y, range_image[nn]]), np.array([x, y, range_image[n]]))

                beta = np.arctan2(d2 * np.sin(phi), d1 - d2 * np.cos(phi))

                if beta > 0.15:
                    q.append(nn)

        q = q[1:]


def find_NNs(segments: torch.Tensor, mask: torch.Tensor):
    # points = segments.nonzero()
    # values = segments[segments > 0]

    points = torch.cat((segments.nonzero(), mask.nonzero()))
    values = torch.cat(
        (segments[segments > 0], torch.zeros(mask.nonzero().shape[0])))

    interpolator = NearestNDInterpolator(points, values)

    X, Y = np.meshgrid(range(segments.shape[0]), range(segments.shape[1]))

    Z = interpolator(X, Y)

    segments = torch.tensor(Z)

    return segments


def neighborhood(point, tans: Quad):
    radius = 25

    neighbors = []
    neighbors = tans.findInRadius(point, radius)

    return neighbors


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
