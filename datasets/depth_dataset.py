import random
from os.path import join

import torch.multiprocessing

from torch.utils.data import Dataset
from utils.data.stego_data_utils import ContrastiveSegDataset

import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from numpy import inf
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from PIL import Image
from torchvision.datasets import VisionDataset, Cityscapes
from torchvision.datasets.utils import verify_str_arg, iterable_to_str, extract_archive
from datasets.cityscapes_depth_dataset import Cityscapes_Depth


class Cityscapes_Depth(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            depth_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(Cityscapes_Depth, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        # self.images_dir = os.path.join(self.root, self.mode, 'leftImg8bit', split) bei allen phaden self.mode eig
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.depth_dir = os.path.join(self.root, "disparity", "disparity_trainvaltest", "disparity", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.depth = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir) or not os.path.isdir(
                self.depth_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
                depth_dir_zip = os.path.join(self.root, 'disparity{}'.format(self.mode, '_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))
                depth_dir_zip = os.path.join(self.root, 'disparity{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))

            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip) and os.path.isfile(depth_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
                extract_archive(from_path=depth_dir_zip, to_path=self.root)

            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            depth_dir = os.path.join(self.depth_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                depth_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0], 'disparity.png')
                self.depth.append(os.path.join(depth_dir, depth_name))
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Union[Any, Any], Union[Any, Any], Union[Any, Any]]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')
        img_d = cv2.imread(self.depth[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256

        disp = img_d
        depth = (0.209313 * 2262.52) / disp
        depth[depth == inf] = 0

        depth_img = Image.fromarray(depth)

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            elif t == 'instance':
                target = cv2.imread(self.targets[index][i], cv2.IMREAD_UNCHANGED).astype(np.uint16)
                target = self._filter_has_instance(target)
                target = self.normalize_labels(target)
                target = Image.fromarray(target)
                targets.append(target)
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, depth_img

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    def normalize_labels(self, instance_labels):
        label_ids = np.unique(instance_labels)
        current_id = 0

        for label_id in label_ids:
            instance_labels = np.where(instance_labels == label_id, current_id, instance_labels)
            current_id += 1

        return instance_labels

    def _filter_has_instance(self, img_array):
        mask = img_array / 1000.0
        img_array = np.where(mask < 24, 0, img_array)
        return img_array


class CityscapesDepth(Dataset):
    def __init__(self, root, image_set, transform, target_transform, depth_transform):
        super(CityscapesDepth, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            self.mode = "fine"
        else:
            our_image_set = image_set
            self.mode = "fine"
        print(root)
        self.inner_loader = Cityscapes_Depth(self.root, our_image_set,
                                             mode=self.mode,
                                             target_type=["semantic", "polygon", "color", "instance"],
                                             transform=None,
                                             target_transform=None,
                                             depth_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.depth_transform = depth_transform
        self.first_nonvoid = 7

    """def __getitem__(self, index):
        if self.transform is not None:
            image, targets, depth_img = self.inner_loader[index]
            depth = np.array(depth_img).astype(np.float32)

            target = targets[0]
            poly = targets[1]
            instance = targets[3]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            real_img = self.depth_transform(image)
            real_img = np.array(real_img).astype('uint8')

            instance = self.depth_transform(instance)
            instance = np.array(instance).astype('uint8')
            image = self.transform(image)

            depth = torch.from_numpy(depth)
            depth = torch.unsqueeze(depth,0)
            depth = torch.unsqueeze(depth, 0)
            depth = F.interpolate(depth, size=(320,640), mode='nearest')
            depth = VF.crop(depth,0,160,320,320)
            depth = torch.squeeze(depth)

            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1

            return image, target, mask, poly, depth, real_img, instance
        else:
            return self.inner_loader[index]

            """

    def __getitem__(self, index):
        if self.transform is not None:
            image, targets, depth = self.inner_loader[index]

            target = targets[0]
            poly = targets[1]
            instance = targets[3]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            real_img = self.depth_transform(image)
            real_img = np.array(real_img).astype('uint8')

            instance = self.depth_transform(instance)
            instance = np.array(instance).astype('uint8')
            image = self.transform(image)
            depth = self.depth_transform(depth)
            depth = np.array(depth).astype('uint8')
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1

            return image, target, mask, poly, depth, real_img, instance
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class ContrastiveDepthDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 depth_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform

        if dataset_name == "cityscapes" and crop_type is None:
            self.n_classes = 27
            dataset_class = CityscapesDepth
            extra_args = dict()
            if image_set == "val":
                extra_args["subset"] = 7
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform,
            depth_transform=depth_transform)

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
        feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            model_type, nice_dataset_name, image_set, crop_type, cfg.res))
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file) or compute_knns:
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        print("path: ", self.dataset[ind])
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        coord_entries = torch.meshgrid([torch.linspace(-1, 1, pack[0].shape[1]),
                                        torch.linspace(-1, 1, pack[0].shape[2])])
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
            "polygons": pack[3],
            "depth": pack[4],
            "real_img": pack[5],
            "instance": pack[6]
        }

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        if self.aug_photometric_transform is not None:
            img_aug = self.aug_photometric_transform(self.aug_geometric_transform(pack[0]))

            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)

            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret
