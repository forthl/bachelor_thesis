import PIL.Image
import cv2
import numpy as np
import torch.multiprocessing
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from PIL import *


def get_assigment(preds,
                  instance):  # there might be less predicted instances than actual ones so make it that it returns correct result

    #preds = normalize_labels(preds)
    #instance = normalize_labels(instance)

    preds = torch.tensor(preds, dtype=torch.int32)
    instance = torch.tensor(instance, dtype=torch.int32)

    num_predicted_instances = int(torch.max(preds).numpy()) + 1
    num_actual_instances = int(torch.max(instance).numpy()) + 1

    if num_predicted_instances > num_actual_instances:
        cost_matrix = torch.zeros(
            (num_actual_instances, num_predicted_instances), dtype=torch.int32)

        preds = preds.reshape(-1)
        instance = instance.reshape(-1)

        total_num = (torch.tensor(num_predicted_instances)) * instance + preds

        cost_matrix = torch.bincount(
            total_num,
            minlength=num_actual_instances * num_predicted_instances)
        cost_matrix = cost_matrix.reshape(
            (num_actual_instances, num_predicted_instances))

        assigments = linear_sum_assignment(cost_matrix, maximize=True)

    else:
        cost_matrix = torch.zeros(
            (num_predicted_instances, num_actual_instances), dtype=torch.int32)

        preds = preds.reshape(-1)
        instance = instance.reshape(-1)

        total_num = (torch.tensor(num_actual_instances)) * preds + instance

        cost_matrix = torch.bincount(
            total_num,
            minlength=num_actual_instances * num_predicted_instances)
        cost_matrix = cost_matrix.reshape(
            (num_predicted_instances, num_actual_instances))
        assigments = linear_sum_assignment(cost_matrix, maximize=True)
        assigments = np.flip(assigments, 0)

    return assigments


def normalize_labels(instances):
    label_ids = np.unique(instances)
    current_id = 0

    for label_id in label_ids:
        instances = np.where(instances == label_id, current_id, instances)
        current_id += 1

    return instances


def get_bounding_boxes(img):
    img = img.astype(np.uint16)
    instance_ids = np.unique(img)
    instance_ids = np.delete(instance_ids, 0)
    bounding_boxes = {}

    for id in instance_ids:  # value 0 are object which have no instances
        indexes = np.nonzero(np.where(img == id, img, 0))
        bounding_boxes[id] = (np.max(indexes[1]) + 1, np.max(indexes[0]) + 1, np.min(indexes[1]) - 1,
                              np.min(indexes[0]) - 1)  # Bounding box encoded as (Right, Bottom, Left, Top)

    return bounding_boxes


# bounding boxes are in array-like coordinate system
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xA - xB, 0)) * max((yA - yB), 0))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[0] - boxA[2]) * (boxA[1] - boxA[3]))
    boxBArea = abs((boxB[0] - boxB[2]) * (boxB[1] - boxB[3]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = interArea / float(boxAArea + boxBArea - interArea + 0.00000001)

    # return the intersection over union value
    return iou


def ap_ar_IoU(target, pred, index):
    target_bin_mask = np.where(target == index, 1, 0)
    pred_bin_mask = np.where(pred == index, 1, 0)

    # PIL.Image.fromarray(np.uint8(target_bin_mask * 255)).show(title="target")
    # PIL.Image.fromarray(np.uint8(pred_bin_mask * 255)).show(title="prediction")

    union = np.add(target_bin_mask, pred_bin_mask)
    intersection = np.where(union == 2, 1, 0)
    union = np.where(union > 0, 1, 0)
    union_sum = union.sum()

    # PIL.Image.fromarray(np.uint8(union * 255)).show(title="union")
    # PIL.Image.fromarray(np.uint8(intersection * 255)).show(title="intersection")

    TP = intersection.sum()
    FP = np.where(pred_bin_mask != intersection, 1, 0).sum()
    FN = np.where(target_bin_mask != intersection, 1, 0).sum()

    if TP == 0:  # so we don't get (NaN, NaN) returned
        return 0, 0, 0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    pixelWise_IoU = TP / union_sum

    return precision, recall, pixelWise_IoU


def get_avg_IoU_AP_AR(target, prediction):
    bounding_boxes_target = get_bounding_boxes(target)
    bounding_boxes_prediction = get_bounding_boxes(prediction)
    B_Box_IoU = []
    precision = []
    recall = []
    pixelIoU = []

    target_keys = np.unique(target)
    prediction_keys = np.unique(prediction)

    all_keys = np.union1d(target_keys, prediction_keys).astype(np.uint8)

    # np.delete(all_keys,0)#this should remove 0 which is backgroung but currently it doesn't

    for key in all_keys:
        if bounding_boxes_prediction.get(key) is not None:
            if bounding_boxes_target.get(key) is not None:
                B_Box_IoU.append(bb_intersection_over_union(bounding_boxes_target[key], bounding_boxes_prediction[key]))
                pr, rc, pixIoU = ap_ar_IoU(target, prediction, key)
                precision.append(pr)
                recall.append(rc)
                pixelIoU.append(pixIoU)
            else:
                B_Box_IoU.append(0)  # if we predicted an instance that is not in the target
                precision.append(0)
                recall.append(0)
                pixelIoU.append(0)
        else:
            B_Box_IoU.append(0)  # if we didn't predict that instance
            precision.append(0)
            recall.append(0)
            pixelIoU.append(0)

    if len(B_Box_IoU) == 0:
        return 0, 0, 0, 0

    A_BBox_IoU = sum(B_Box_IoU) / len(B_Box_IoU)
    AP = sum(precision) / len(precision)
    AR = sum(recall) / len(recall)
    A_pixel_IoU = sum(pixelIoU) / len(pixelIoU)

    return A_BBox_IoU, AP, AR, A_pixel_IoU, B_Box_IoU, precision, recall, pixelIoU


def drawBoundingBoxes(imageData, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    imgHeight, imgWidth, _ = imageData.shape
    for res in inferenceResults:
        right = min(int(res[0]), imgWidth)
        bottom = min(int(res[1]), imgHeight)
        left = max(int(res[2]), 0)
        top = max(int(res[3]), 0)
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData, (left, top), (right, bottom), color, thick)

    return imageData