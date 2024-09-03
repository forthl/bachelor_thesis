import collections
import os
from os.path import join
import io

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchmetrics import Metric
from torchvision import models
from torchvision import transforms as T
from torch.utils.tensorboard.summary import hparams


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = T.ToTensor()(image)
    writer.add_image(name, image, step)
    plt.clf()
    plt.close()


@torch.jit.script
def shuffle(x):
    return x[torch.randperm(x.shape[0])]


def add_hparams_fixed(writer, hparam_dict, metric_dict, global_step):
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_step)


@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


def one_hot_feats(labels, n_classes):
    return F.one_hot(labels, n_classes).permute(0, 3, 1, 2).to(torch.float32)


def load_model(model_type, data_dir):
    if model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download",
                          model_file)
        model_weights = torch.load(model_file)
        # model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
        #                          'model' in name}
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    model.eval()
    model.cuda()
    return model


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def get_transform(res, is_label, crop_type):
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
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])





def filter_big_classes(class_masks, classes):
    num_classes = len(classes)
    threshold = 1000 / (5/num_classes)
    class_masks_filtered = []
    classes_filtered = []
    for i in range(num_classes):
        if np.sum(class_masks[i]) > threshold: #check if the class has more than threshold pixels
            class_masks_filtered.append(class_masks[i])
            classes_filtered.append(classes[i])
    if len(class_masks_filtered) == 0:
        class_masks_filtered = class_masks
        classes_filtered = classes
    return class_masks_filtered, classes_filtered



def filter_mask(mask):
    mask = filter_classes_has_instance(mask)
    return mask





def filter_classes_has_instance(mask):
    image_shape = mask.shape
    has_instance_list = [
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
       # (0, 0, 90),
       # (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if not np.any(np.all(mask[i, j] == has_instance_list, axis=1)):
                mask[i, j] = [0, 0, 0]

    return mask


def write_results(result_dir, index, Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU, B_Box_IoU, precision, recall, pixelIoU):

    result_dir = join(result_dir, "Metrics/")

    f = open(join(result_dir, "BBox_IoU.txt"), "a")
    f.write(str(index)+":      "+str(Avg_BBox_IoU) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "AP.txt"), "a")
    f.write(str(index)+":      "+str(AP) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "AR.txt"), "a")
    f.write(str(index)+":      "+str(AR) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "Pixel_IoU.txt"), "a")
    f.write(str(index)+":      "+str(Avg_Pixel_IoU) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "BBox_IoU_Individual_instance.txt"), "a")
    f.write(str(index) + ":      ")
    for val in B_Box_IoU:
        f.write(str(val) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "precision_Individual_instance.txt"), "a")
    f.write(str(index) + ":      ")
    for val in precision:
        f.write(str(val) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "recall_Individual_instance.txt"), "a")
    f.write(str(index) + ":      ")
    for val in recall:
        f.write(str(val) + " , ")
    f.write("\n")
    f.close()

    f = open(join(result_dir, "Pixel_IoU_Individual_instance.txt"), "a")
    f.write(str(index) + ":      ")
    for val in pixelIoU:
        f.write(str(val) + " , ")
    f.write("\n")
    f.close()


def write_images(result_dir, index, real_image, segmentation_mask, predicted_segmentation_mask, instance_mask,
                 predicted_instance_mask, bounding_box):
    real_image.save(join(result_dir, "real_img" , str(index) + ".png"))
    segmentation_mask.save(join(result_dir, "semantic_target", str(index) + ".png"))
    predicted_segmentation_mask.save(join(result_dir, "semantic_predicted", str(index) + ".png"))
    instance_mask.save(join(result_dir, "instance_target", str(index) + ".png"))
    predicted_instance_mask.save(join(result_dir, "instance_predicted", str(index) + ".png"))
    bounding_box.save(join(result_dir, "bounding_boxes", str(index) + ".png"))

def resize_mask(mask, size):
    mask = torch.tensor(mask.astype('float32'))
    if mask.ndim == 3:
        mask = torch.unsqueeze(mask,0)
    mask = mask.permute((0, 3, 1, 2))

    mask = F.interpolate(input=mask, size=size, mode='bilinear', align_corners=False)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask.numpy()

    plotted_img = Image.fromarray(mask[0].astype(np.uint8))
    #plotted_img.show()
    plotted_array = np.array(plotted_img)

    # Use matplotlib to display the image
    plt.imshow(plotted_array)
    plt.show()

    return mask


def grayscale_to_random_color(grayscale, image_shape, color_list):
    result = np.zeros((image_shape[0], image_shape[1], 3))
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            meyer = int(grayscale[i, j])
            if meyer < len(color_list):
                result[i, j] = color_list[meyer]
    return result



def get_depth_transform(res,  crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))

    return T.Compose([T.Resize(res, Image.NEAREST),cropper]) #should be (res,res) but it works this way, I don't know why


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1).cuda()
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))



random_colors = [[0,0,0],
    [242, 30, 119],
[180, 78, 5],
[172, 65, 58],
[230, 44, 107],
[75, 138, 110],
[43, 154, 217],
[103, 119, 75],
[95, 124, 140],
[68, 42, 239],
[198, 8, 208],
[107, 31, 86],
[112, 225, 133],
[180, 142, 95],
[137, 148, 188],
[191, 131, 237],
[217, 175, 19],
[12, 111, 82],
[225, 98, 68],
[244, 193, 169],
[15, 237, 77],
[109, 241, 98],
[172, 2, 232],
[87, 12, 137],
[169, 80, 95],
[245, 236, 37],
[150, 10, 181],
[52, 184, 121],
[17, 39, 47],
[177, 221, 223],
[45, 248, 151],
[141, 26, 228],
[27, 61, 138],
[127, 52, 34],
[24, 109, 132],
[186, 29, 106],
[60, 102, 222],
[143, 70, 122],
[83, 194, 80],
[101, 131, 243],
[10, 62, 26],
[22, 219, 3],
[28, 86, 100],
[61, 6, 47],
[10, 185, 33],
[201, 201, 82],
[70, 151, 48],
[150, 77, 129],
[246, 241, 171],
[99, 164, 92],
[108, 225, 254],
[155, 0, 115],
[227, 212, 230],
[2, 82, 203],
[196, 150, 77],
[251, 46, 154],
[55, 67, 253],
[14, 116, 174],
[4, 232, 192],
[158, 138, 81],
[87, 136, 161],
[40, 40, 164],
[11, 54, 225],
[195, 125, 236],
[158, 240, 162],
[54, 149, 239],
[244, 164, 149],
[248, 54, 168],
[103, 134, 150],
[218, 4, 139],
[160, 182, 186],
[51, 139, 50],
[54, 23, 69],
[62, 57, 66],
[157, 152, 54],
[231, 174, 129],
[89, 52, 117],
[29, 131, 84],
[161, 9, 12],
[24, 178, 168],
[245, 37, 72],
[135, 175, 198],
[53, 150, 148],
[4, 1, 55],
[250, 16, 196],
[178, 64, 161],
[187, 91, 100],
[202, 86, 219],
[3, 1, 209],
[154, 154, 230],
[166, 49, 118],
[249, 18, 35],
[110, 154, 6],
[188, 10, 131],
[35, 174, 163],
[144, 35, 124],
[8, 96, 112],
[109, 37, 182],
[95, 212, 19],
[249, 183, 108],
[107, 201, 190],
[51, 239, 83],
[254, 198, 173],
[44, 237, 198],
[252, 244, 234],
[233, 194, 181],
[57, 170, 93],
[249, 138, 45],
[166, 140, 130],
[165, 71, 171],
[2, 181, 65],
[139, 194, 73],
[164, 191, 76],
[167, 80, 239],
[95, 77, 125],
[62, 164, 228],
[35, 76, 174],
[12, 70, 49],
[125, 77, 56],
[216, 225, 153],
[241, 95, 93],
[75, 101, 142],
[179, 36, 160],
[57, 64, 219],
[105, 58, 192],
[208, 255, 38],
[14, 14, 144],
[121, 0, 91],
[215, 243, 160],
[73, 63, 124],
[96, 23, 101],
[41, 252, 59],
[179, 229, 141],
[126, 152, 69],
[231, 9, 201],
[250, 41, 168],
[214, 13, 174],
[208, 14, 188],
[189, 245, 150],
[58, 154, 186],
[105, 189, 58],
[126, 102, 202],
[77, 157, 145],
[165, 28, 19],
[50, 142, 137],
[132, 197, 131],
[39, 219, 13],
[40, 235, 252],
[26, 184, 249],
[216, 33, 144],
[214, 196, 155],
[191, 188, 231],
[248, 246, 118],
[44, 103, 149],
[65, 241, 166],
[139, 219, 107],
[181, 11, 237],
[117, 168, 13],
[129, 7, 183],
[102, 148, 90],
[148, 204, 138],
[55, 99, 214],
[95, 69, 109],
[125, 243, 113],
[166, 241, 73],
[52, 95, 52],
[159, 114, 48],
[189, 58, 23],
[24, 178, 190],
[17, 237, 8],
[200, 123, 98],
[251, 196, 242],
[62, 7, 189],
[5, 249, 95],
[244, 8, 250],
[46, 135, 181],
[239, 11, 36],
[133, 175, 2],
[20, 174, 68],
[175, 117, 40],
[224, 132, 217],
[135, 216, 194],
[181, 83, 158],
[142, 216, 102],
[68, 76, 235],
[215, 111, 190],
[41, 133, 187],
[181, 199, 248],
[208, 94, 39],
[15, 115, 63],
[139, 227, 253],
[159, 142, 235],
[144, 57, 198],
[60, 63, 167],
[129, 117, 169],
[223, 204, 99],
[47, 67, 122],
[243, 254, 28],
[191, 252, 182],
[224, 50, 216],
[12, 125, 53],
[108, 46, 99],
[132, 154, 179],
[33, 134, 64],
[144, 165, 79],
[72, 203, 68],
[91, 107, 54],
[58, 198, 32],
[173, 147, 246],
[21, 206, 120],
[219, 212, 193],
[50, 238, 11],
[135, 167, 48],
[182, 165, 120],
[111, 30, 171],
[183, 69, 46],
[125, 166, 20],
[194, 41, 190],
[156, 160, 54],
[218, 140, 185],
[48, 174, 70],
[169, 248, 44],
[252, 224, 11],
[128, 3, 240],
[155, 143, 215],
[82, 220, 32],
[99, 84, 238],
[77, 254, 21],
[103, 227, 205],
[60, 54, 197],
[159, 74, 68],
[60, 174, 144],
[116, 166, 162],
[7, 37, 25],
[206, 140, 72],
[110, 201, 254],
[181, 216, 43],
[194, 135, 170],
[131, 109, 104],
[113, 57, 244],
[228, 185, 179],
[197, 161, 0],
[133, 3, 119],
[211, 176, 33],
[31, 82, 0],
[31, 30, 163],
[113, 209, 116],
[46, 172, 141],
[238, 100, 210],
[149, 255, 207],
[135, 105, 142],
[118, 230, 47],
[112, 74, 176],
[148, 13, 70],
[163, 72, 70],
[56, 104, 14],
[140, 49, 132],
[25, 215, 185],
[52, 129, 229],
[93, 127, 73],
[91, 226, 112],
[194, 17, 34],
[246, 248, 128],
[142, 244, 247],
[34, 5, 197],
[0, 152, 90],
[48, 126, 97],
[47, 95, 236],
[3, 112, 213],
[124, 196, 124],
[179, 6, 186],
[22, 95, 116],
[97, 88, 31],
[142, 236, 152],
[158, 35, 14],
[135, 155, 100],
[0, 82, 247],
[181, 125, 128],
[11, 217, 252],
[245, 86, 135],
[183, 27, 38],
[160, 14, 181],
[172, 173, 186],
[177, 35, 80],
[29, 183, 44],
[13, 8, 38],
[66, 116, 29],
[252, 195, 153],
[179, 204, 249],
[127, 91, 160],
[60, 204, 180],
[12, 61, 154],
[204, 237, 16],
[165, 1, 48],
[149, 185, 100],
[85, 49, 243],
[211, 154, 102],
[237, 182, 61],
[201, 216, 254],
[191, 23, 54],
[175, 120, 24],
[162, 251, 187],
[145, 237, 84],
[87, 210, 75],
[235, 42, 75],
[219, 179, 123],
[111, 31, 214],
[111, 99, 126],
[128, 234, 52],
[29, 107, 109],
[244, 57, 251],
[191, 27, 129],
[229, 221, 116],
[250, 250, 50],
[178, 58, 192],
[165, 95, 153],
[15, 189, 21],
[127, 100, 174],
[119, 138, 222],
[157, 229, 93],
[232, 149, 99],
[229, 235, 113],
[252, 133, 61],
[43, 206, 60],
[126, 194, 151],
[230, 178, 227],
[137, 8, 14],
[179, 9, 174],
[140, 67, 8],
[55, 199, 100],
[152, 11, 234],
[13, 158, 131],
[234, 102, 109],
[166, 130, 170],
[242, 114, 187],
[98, 167, 197],
[209, 5, 138],
[219, 109, 17],
[89, 171, 115],
[176, 118, 40],
[176, 181, 50],
[91, 180, 63],
[119, 70, 32],
[37, 162, 183],
[8, 20, 173],
[18, 169, 202],
[181, 55, 115],
[95, 189, 195],
[158, 249, 218],
[74, 116, 77],
[143, 235, 205],
[120, 139, 232],
[97, 194, 121],
[239, 156, 254],
[204, 188, 123],
[19, 99, 62],
[198, 121, 182],
[254, 201, 202],
[251, 129, 137],
[172, 188, 252],
[223, 195, 228],
[16, 227, 86],
[224, 95, 233],
[255, 56, 27],
[183, 254, 170],
[147, 194, 167],
[242, 74, 111],
[2, 234, 189],
[12, 222, 217],
[226, 0, 80],
[234, 212, 160],
[108, 169, 34],
[231, 156, 9],
[124, 33, 19],
[150, 107, 188],
[27, 94, 31],
[192, 235, 61],
[198, 54, 46],
[82, 119, 86],
[31, 211, 171],
[194, 31, 103],
[184, 74, 49],
[158, 100, 9],
[242, 164, 97],
[65, 234, 116],
[243, 112, 37],
[78, 88, 222],
[209, 117, 153],
[55, 13, 209],
[96, 100, 179],
[134, 92, 21],
[41, 36, 47],
[199, 52, 19],
[81, 183, 171],
[247, 134, 150],
[98, 130, 242],
[154, 164, 129],
[131, 159, 150],
[25, 16, 209],
[185, 237, 85],
[254, 216, 117],
[209, 204, 179],
[42, 152, 196],
[146, 241, 5],
[149, 3, 68],
[28, 39, 113],
[224, 212, 139],
[194, 39, 106],
[165, 52, 99],
[172, 214, 92],
[16, 57, 134],
[242, 175, 185],
[173, 14, 33],
[220, 243, 130],
[87, 3, 84],
[54, 40, 72],
[94, 9, 185],
[93, 104, 82],
[169, 36, 77],
[227, 133, 203],
[249, 121, 179],
[208, 9, 38],
[220, 49, 33],
[89, 159, 227],
[172, 241, 104],
[123, 73, 213],
[5, 202, 117],
[59, 35, 18],
[72, 173, 201],
[203, 20, 43],
[110, 127, 113],
[169, 179, 218],
[181, 160, 154],
[203, 156, 106],
[193, 247, 29],
[145, 75, 7],
[152, 178, 223],
[118, 193, 240],
[181, 81, 37],
[122, 1, 131],
[16, 32, 57],
[84, 66, 13],
[126, 220, 238],
[67, 233, 173],
[71, 44, 172],
[159, 190, 223],
[166, 210, 113],
[157, 234, 76],
[54, 55, 109],
[253, 222, 182],
[165, 226, 67],
[17, 109, 151],
[39, 180, 182],
[162, 198, 39],
[31, 16, 109],
[72, 79, 204],
[162, 62, 86],
[113, 196, 250],
[162, 231, 49],
[242, 205, 94],
[50, 169, 253],
[30, 99, 182],
[50, 167, 67],
[175, 176, 19],
[98, 95, 208],
[142, 42, 26],
[219, 114, 64],
[225, 51, 167],
[190, 72, 237],
[120, 199, 83],
[250, 85, 100],
[181, 213, 46],
[152, 71, 241],
[94, 233, 176],
[8, 47, 226],
[0, 185, 100],
[238, 152, 37],
[20, 231, 93],
[16, 41, 175],
[9, 186, 176],
[38, 193, 10],
[13, 73, 57],
[203, 52, 227],
[108, 106, 13],
[68, 220, 129],
[19, 76, 86],
[177, 86, 20],
[71, 39, 68],
[131, 148, 113],
[80, 114, 58],
[146, 102, 159],
[57, 219, 129],
[242, 74, 232],
[252, 216, 127],
[147, 22, 22],
[3, 227, 129],
[58, 22, 48],
[92, 255, 241],
[150, 54, 92],
[162, 80, 153],
[67, 214, 107],
[214, 0, 90],
[41, 29, 44],
[56, 53, 217],
[155, 115, 102],
[204, 12, 254],
[218, 176, 145],
[190, 136, 73],
[142, 230, 65],
[115, 153, 87],
[120, 245, 43],
[139, 1, 80],
[169, 160, 159],
[20, 198, 9],
[61, 75, 53],
[70, 95, 34],
[108, 52, 87],
[213, 116, 45],
[200, 175, 82],
[54, 133, 226],
[212, 154, 208],
[135, 59, 62],
[74, 244, 78],
[118, 28, 138],
[66, 55, 33],
[138, 145, 209],
[195, 226, 230],
[74, 142, 116],
[17, 108, 147],
[72, 55, 112],
[71, 174, 13],
[171, 194, 219],
[195, 68, 35],
[167, 232, 126],
[238, 49, 62],
[117, 61, 237],
[219, 195, 31],
[80, 130, 162],
[130, 5, 247],
[10, 77, 117],
[181, 99, 253],
[134, 45, 7],
[159, 233, 85],
[181, 196, 156],
[19, 99, 87],
[144, 209, 178],
[153, 111, 141],
[117, 130, 81],
[54, 85, 89],
[161, 73, 200],
[142, 172, 193],
[51, 208, 108],
[26, 95, 208],
[190, 51, 243],
[239, 182, 12],
[96, 76, 87],
[124, 204, 231],
[236, 208, 235],
[176, 247, 212],
[68, 113, 113],
[95, 125, 233],
[187, 180, 194],
[200, 99, 208],
[66, 91, 177],
[94, 76, 106],
[190, 12, 11],
[185, 7, 93],
[230, 63, 67],
[144, 175, 83],
[236, 79, 43],
[140, 77, 189],
[238, 31, 161],
[112, 21, 18],
[58, 154, 238],
[7, 191, 237],
[28, 149, 160],
[118, 112, 99],
[138, 162, 83],
[209, 170, 236],
[216, 224, 244],
[145, 242, 86],
[40, 161, 114],
[219, 246, 34],
[24, 149, 3],
[181, 132, 4],
[21, 121, 252],
[187, 222, 186],
[48, 0, 234],
[182, 173, 40],
[29, 235, 174],
[89, 78, 108],
[33, 232, 120],
[91, 97, 18],
[64, 28, 189],
[223, 5, 250],
[102, 100, 117],
[240, 251, 206],
[156, 42, 136],
[43, 208, 205],
[67, 164, 124],
[133, 28, 152],
[0, 43, 229],
[93, 81, 4],
[61, 32, 136],
[6, 231, 170],
[107, 45, 29],
[192, 159, 28],
[245, 97, 135],
[231, 111, 59],
[226, 180, 179],
[142, 143, 43],
[53, 59, 30],
[232, 77, 161],
[243, 191, 218],
[3, 204, 82],
[179, 99, 213],
[34, 133, 125],
[231, 28, 47],
[82, 16, 25],
[23, 115, 193],
[229, 191, 4],
[100, 14, 118],
[1, 95, 235],
[35, 164, 198],
[24, 0, 52],
[138, 184, 62],
[218, 24, 155],
[251, 89, 156],
[66, 198, 87],
[150, 230, 113],
[19, 146, 28],
[30, 29, 113],
[58, 173, 47],
[194, 201, 214],
[71, 190, 112],
[162, 254, 152],
[249, 150, 186],
[245, 70, 198],
[213, 194, 71],
[217, 153, 226],
[207, 76, 166],
[13, 25, 129],
[206, 178, 181],
[57, 140, 100],
[190, 48, 183],
[254, 245, 113],
[255, 35, 168],
[252, 78, 11],
[187, 134, 62],
[148, 214, 39],
[38, 252, 160],
[88, 238, 41],
[206, 183, 139],
[90, 36, 111],
[168, 65, 193],
[179, 135, 148],
[236, 204, 40],
[63, 227, 0],
[88, 253, 138],
[61, 232, 154],
[8, 99, 76],
[214, 194, 29],
[140, 139, 75],
[121, 152, 136],
[234, 250, 72],
[250, 66, 75],
[94, 16, 111],
[158, 8, 45],
[170, 69, 131],
[68, 90, 135],
[214, 28, 81],
[184, 167, 136],
[183, 140, 124],
[92, 73, 188],
[3, 13, 187],
[181, 15, 183],
[234, 199, 105],
[216, 178, 225],
[133, 42, 130],
[84, 53, 207],
[102, 100, 134],
[88, 226, 127],
[135, 20, 249],
[231, 253, 245],
[181, 9, 226],
[211, 25, 224],
[119, 222, 201],
[254, 176, 99],
[228, 71, 52],
[85, 56, 137],
[209, 235, 175],
[42, 110, 95],
[104, 118, 1],
[77, 89, 134],
[145, 20, 42],
[212, 191, 228],
[239, 99, 255],
[250, 115, 95],
[232, 183, 122],
[220, 193, 162],
[138, 174, 33],
[179, 218, 31],
[223, 20, 108],
[127, 16, 102],
[184, 155, 226],
[83, 242, 168],
[0, 54, 43],
[208, 224, 38],
[230, 204, 240],
[84, 180, 224],
[149, 243, 57],
[21, 109, 79],
[12, 15, 32],
[51, 48, 73],
[158, 229, 179],
[161, 152, 187],
[236, 95, 194],
[202, 141, 182],
[38, 246, 223],
[203, 143, 208],
[216, 194, 22],
[164, 92, 150],
[183, 48, 41],
[127, 218, 236],
[148, 44, 34],
[64, 228, 78],
[43, 104, 17],
[231, 151, 12],
[243, 229, 130],
[156, 95, 129],
[185, 229, 76],
[230, 167, 77],
[37, 116, 93],
[254, 137, 183],
[228, 56, 100],
[146, 170, 38],
[3, 56, 82],
[227, 144, 148],
[115, 249, 59],
[38, 198, 244],
[214, 133, 195],
[222, 80, 32],
[241, 205, 45],
[98, 250, 71],
[90, 20, 236],
[21, 126, 108],
[228, 18, 51],
[80, 200, 25],
[146, 39, 179],
[160, 37, 47],
[76, 56, 228],
[245, 30, 14],
[152, 220, 181],
[200, 5, 10],
[138, 164, 209],
[234, 200, 136],
[13, 162, 107],
[219, 107, 26],
[3, 32, 17],
[122, 208, 235],
[63, 76, 178],
[41, 14, 241],
[245, 234, 53],
[248, 1, 35],
[63, 43, 115],
[37, 150, 253],
[222, 188, 122],
[151, 148, 232],
[154, 5, 67],
[10, 232, 0],
[66, 114, 237],
[49, 3, 67],
[144, 84, 248],
[196, 63, 184],
[19, 164, 166],
[163, 7, 53],
[155, 6, 29],
[3, 56, 169],
[167, 91, 5],
[194, 109, 211],
[18, 177, 236],
[66, 77, 230],
[123, 71, 189],
[201, 153, 236],
[87, 112, 107],
[49, 182, 156],
[113, 232, 114],
[80, 129, 226],
[2, 219, 227],
[124, 218, 52],
[1, 89, 166],
[27, 215, 33],
[244, 247, 104],
[166, 77, 80],
[81, 5, 142],
[37, 232, 162],
[54, 76, 92],
[149, 65, 107],
[231, 148, 194],
[187, 161, 6],
[112, 143, 127],
[81, 45, 208],
[161, 174, 167],
[44, 243, 39],
[123, 53, 89],
[84, 135, 99],
[206, 84, 162],
[187, 57, 231],
[118, 231, 164],
[99, 32, 14],
[204, 155, 220],
[139, 163, 53],
[120, 6, 242],
[165, 118, 113],
[2, 33, 155],
[245, 243, 105],
[26, 151, 242],
[5, 5, 145],
[145, 224, 49],
[244, 38, 117],
[47, 141, 63],
[110, 157, 75],
[216, 46, 120],
[183, 84, 43],
[228, 121, 44],
[214, 206, 169],
[38, 225, 228],
[218, 238, 223],
[34, 63, 142],
[2, 89, 124],
[114, 89, 6],
[56, 64, 146],
[39, 35, 242],
[171, 218, 124],
[175, 23, 190],
[127, 197, 10],
[171, 166, 246],
[40, 21, 33],
[116, 79, 173],
[31, 204, 211],
[161, 203, 173],
[221, 52, 160],
[189, 184, 254],
[70, 50, 160],
[111, 225, 94],
[95, 39, 140],
[222, 220, 19],
[206, 82, 237],
[168, 92, 156],
[23, 14, 192],
[252, 76, 164],
[167, 111, 56],
[164, 237, 218],
[27, 95, 30],
[142, 72, 181],
[165, 137, 76],
[42, 129, 211],
[198, 76, 128],
[142, 41, 13],
[123, 8, 195],
[40, 169, 21],
[197, 200, 11],
[192, 89, 105],
[125, 66, 116],
[116, 57, 157],
[151, 89, 137],
[33, 188, 18],
[197, 177, 55],
[77, 243, 56],
[151, 243, 49],
[160, 116, 172],
[1, 222, 201],
[37, 34, 41],
[207, 103, 183],
[27, 248, 251],
[133, 95, 138],
[254, 171, 9],
[104, 97, 133],
[64, 34, 142],
[110, 154, 120],
[91, 245, 134],
[114, 174, 45],
[210, 143, 85],
[253, 245, 176],
[131, 17, 65],
[26, 142, 89],
[175, 194, 118],
[187, 162, 78],
[172, 33, 184],
[84, 173, 182],
[156, 21, 122],
[234, 141, 140],
[233, 82, 226],
[177, 27, 251],
[70, 62, 174],
[94, 199, 170],
[34, 216, 137],
[107, 124, 42],
[143, 76, 144],
[125, 7, 138],
[63, 234, 229],
[19, 168, 248],
[16, 31, 227],
[253, 249, 212],
[74, 30, 56],
[197, 28, 214],
[180, 70, 100],
[183, 133, 114],
[91, 14, 101],
[153, 197, 227],
[42, 140, 31],
[105, 118, 138],
[80, 67, 37],
[28, 1, 235],
[20, 219, 222],
[246, 88, 142],
[39, 139, 213],
[248, 15, 119],
[233, 54, 21],
[85, 38, 248],
[75, 12, 0],
[21, 99, 42],
[207, 158, 177],
[154, 126, 92],
[71, 90, 26],
[219, 150, 244],
[55, 29, 249],
[123, 38, 243],
[245, 33, 200],
[211, 247, 239],
[65, 154, 21],
[186, 236, 9],
[238, 239, 251],
[144, 123, 26],
[14, 110, 111],
[132, 98, 16],
[168, 247, 121],
[0, 250, 64],
[28, 82, 249],
[75, 36, 194],
[247, 24, 87],
[107, 102, 88],
[159, 17, 155],
[90, 190, 126],
[82, 208, 140],
[210, 230, 38],
[127, 63, 235],
[6, 237, 24],
[203, 135, 28],
[3, 194, 97],
[230, 249, 230],
[111, 58, 31],
[30, 196, 15],
[20, 24, 255],
[170, 177, 70],
[101, 255, 76],
[121, 244, 109],
[18, 21, 34],
[120, 15, 190],
[8, 116, 133],
[162, 107, 40],
[207, 150, 57],
[4, 217, 31],
[171, 130, 73],
[56, 239, 224],
[133, 44, 62],
[2, 222, 234],
[80, 24, 128],
[245, 249, 230],
[91, 141, 179],
[246, 125, 92],
[248, 9, 123],
[141, 253, 228],
[196, 205, 8],
[222, 141, 116],
[95, 25, 111],
[240, 233, 119],
[110, 199, 133],
[6, 243, 139],
[194, 196, 195],
[148, 196, 150],
[90, 246, 130],
[160, 164, 36],
[104, 62, 70],
[252, 152, 102],
[82, 247, 179],
[24, 41, 65],
[7, 36, 177],
[252, 192, 224],
[153, 57, 208],
[148, 170, 188],
[16, 126, 23],
[236, 215, 108],
[144, 129, 31],
[146, 164, 89],
[177, 204, 174],
[8, 72, 195],
[172, 142, 229],
[232, 210, 130],
[36, 83, 72],
[249, 73, 85]]