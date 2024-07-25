import cv2

import numpy as np

from PIL import Image
from torchvision import transforms as T


def getBinaryMasks(data, instanceClasses):
    polygons = data.get("objects")

    binary_masks = {}

    for className in instanceClasses:
        binary_masks[className] = []

    for objects in polygons:
        label = objects.get("label")[0]
        if label in instanceClasses:
            area = objects.get("polygon")
            area = [[x[0].item(), x[1].item()] for x in area]
            area = np.asarray(area)
            filled = np.zeros([1024, 2048])
            filled = Image.fromarray(cv2.fillPoly(filled, pts=[area], color=(255, 255, 255)))
            transform = get_Image_transform(320, False, "center")
            filled = transform(filled)
            if not (cv2.countNonZero(np.array(filled)) == 0):
                binary_masks[label].append(filled)

    return binary_masks


def get_Image_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper])


def iou(mask1, mask2):
    mask1 = np.asarray(mask1) / 255
    mask2 = np.asarray(mask2) / 255

    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union
