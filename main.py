import numpy as np  # dont remove this otherwise gets stuck in infinite loop
import os
from os.path import join
from datasets.depth_dataset import ContrastiveDepthDataset
from utils.eval_segmentation import batched_crf
from utils.eval_segmentation import _apply_crf
from utils.eval_segmentation import dense_crf
from utils.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import utils.maskDepth2 as maskD
from utils.train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
from utils.utils_folder.stego_utils import get_depth_transform, get_transform
import torchvision.transforms as T
import utils.evaluation_utils as eval_utils
from utils.drive_seg_geo_transformation import labelRangeImage
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, recall_score
from matplotlib import pyplot as plt
from multiprocessing import Pool, Manager, Process

torch.multiprocessing.set_sharing_strategy('file_system')


def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask

    return_dict[procnum] = labelRangeImage(rel_depth)


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_directory_path = cfg.results_dir
    result_dir = join(result_directory_path, "results/predictions/geo_transformation_good_images/")
    os.makedirs(join(result_dir, "1_1", "Metrics"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "real_img"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "semantic_target"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "semantic_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "instance_target"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "instance_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "1_1", "bounding_boxes"), exist_ok=True)
    os.makedirs(join(result_dir, "Metrics"), exist_ok=True)
    os.makedirs(join(result_dir, "real_img"), exist_ok=True)
    os.makedirs(join(result_dir, "semantic_target"), exist_ok=True)
    os.makedirs(join(result_dir, "semantic_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "instance_target"), exist_ok=True)
    os.makedirs(join(result_dir, "instance_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "bounding_boxes"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

    color_list = []
    # 0 values are not part of any instance thus black
    color_list.append((0, 0, 0))

    for i in range(1000):
        color = list(np.random.choice(range(256), size=3))
        color_list.append(color)

    depth_transform_res = cfg.res

    if cfg.resize_to_original:
        depth_transform_res = cfg.resize_res

    loader_crop = "center"
    image_shape = (depth_transform_res, depth_transform_res)

    test_dataset = ContrastiveDepthDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.experiment_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, loader_crop),
        target_transform=get_transform(cfg.res, True, loader_crop),
        depth_transform=get_depth_transform(depth_transform_res, loader_crop),
        cfg=model.cfg,
    )

    loader = DataLoader(test_dataset, cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    # TODO Try to patch the image into 320x320 and then feed it into the transformer

    count_naming = 0

    for i, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = batch["img"].cuda()
            label = batch["label"].cuda()
            depth = batch["depth"]
            real_img = batch["real_img"]
            instance = batch["instance"]

            transToImg = T.ToPILImage()
            real_img = real_img[0]
            instance = instance[0].numpy()
            instance = eval_utils.normalize_labels(instance)
            instance_img = grayscale_to_random_color(
                instance, image_shape, color_list).astype('uint8')
            depth_img = transToImg(depth[0])

            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(
                code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(
                model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(
                code, 2, log_probs=True).cpu()

            # -----------------------------
            # non crf cluster predictions
            # -----------------------------

            # cluster_preds = cluster_probs.argmax(1)

            # --------------------------------------------
            # workaround for batch crf as pool.map won't work on my PC
            # --------------------------------------------
            res = []
            for re in map(_apply_crf, zip(img.detach().cpu(), cluster_probs.detach().cpu())):
                res.append(re)

            res = np.array(res)
            cluster_preds = torch.cat([torch.from_numpy(arr).unsqueeze(
                0) for arr in res], dim=0).argmax(1).cuda()

            # --------------
            # batched crf
            # ------------
            # cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()

            model.test_cluster_metrics.update(cluster_preds, label)

            tb_metrics = {
                **model.test_linear_metrics.compute(),
                **model.test_cluster_metrics.compute(), }

            plotted = model.label_cmap[model.test_cluster_metrics.map_clusters(
                cluster_preds.cpu())].astype(np.uint8)
            # Assuming 'plotted' is already in the correct shape [H, W, C] and dtype=np.uint8
            # Convert the first image in the batch to a format that can be displayed
            plotted_img = Image.fromarray(plotted[0])

            # Convert the PIL Image back to a NumPy array for plotting
            plotted_array = np.array(plotted_img)

            # Use matplotlib to display the image
            plt.imshow(plotted_array)
            plt.show()
            # plotted_img = Image.fromarray(plotted[0])
            # plotted_img.show()
            plotted_filtered = filter_classes_has_instance(
                plotted[0])  # sets backgroung to 0
            plotted_img = Image.fromarray(
                plotted_filtered.astype(np.uint8))
            # plotted_img.show()
            plotted_array = np.array(plotted_img)

            # Use matplotlib to display the image
            plt.imshow(plotted_array)
            plt.show()

            if cfg.resize_to_original:
                plotted_filtered = resize_mask(plotted_filtered, image_shape)
                plotted_img = Image.fromarray(plotted_filtered[0].astype(np.uint8))

            masks = maskD.get_segmentation_masks(plotted_img)
            # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road buildingi)
            masks.pop(0)
            # masked_depths = maskD.get_masked_depth(depth_img, masks)

            depth_array = np.asarray(depth_img)
            depth_array = np.array(
                256 * depth_array / 0x0fff, dtype=np.float32)

            manager = Manager()
            return_dict = manager.dict()
            jobs = []
            for i in range(len(masks)):
                p = Process(target=worker, args=(
                    i, return_dict, depth_array, masks[i]))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            current_num_instances = 0
            predicted_instance_mask = np.zeros(masks[0].shape)
            # fig, axeslist = plt.subplots(ncols=3, nrows=3)

            for k in return_dict.keys():
                labels = len(np.unique(return_dict[k])) - 1
                instance_mask = return_dict[k]

                instance_mask = np.where(instance_mask != 0, instance_mask + current_num_instances, 0)
                current_num_instances += labels

                predicted_instance_mask = np.add(predicted_instance_mask, instance_mask)

            predicted_instance_mask = eval_utils.normalize_labels(predicted_instance_mask)
            instance = eval_utils.normalize_labels(instance)

            instance_mask_not_matched = np.zeros(image_shape)

            predicted_instance_ids = np.unique(predicted_instance_mask)

            assignments = eval_utils.get_assigment(predicted_instance_mask,
                                                   instance)

            num_matched_instances = assignments[0].size

            not_matched_instance_ids = np.setdiff1d(predicted_instance_ids, assignments[1])

            instance_mask_matched = np.zeros(image_shape)

            for i, val in enumerate(assignments[1]):
                mask = np.where(predicted_instance_mask ==
                                val, assignments[0][i], 0)
                instance_mask_matched = instance_mask_matched + mask

            for i, id in enumerate(not_matched_instance_ids):
                instance_mask_not_matched = np.add(instance_mask_not_matched,
                                                   np.where(predicted_instance_mask == id,
                                                            num_matched_instances + i,
                                                            0))

            #instance_mask_matched_N_M = np.add(instance_mask_matched, instance_mask_not_matched)

            #instance_mask_predicted_N_M = Image.fromarray(
                #grayscale_to_random_color(instance_mask_matched_N_M, image_shape, color_list).astype(np.uint8))
            instance_mask_predicted_1_1 = Image.fromarray(
                grayscale_to_random_color(instance_mask_matched, image_shape, color_list).astype(np.uint8))
            instance_mask_predicted_1_1_array = np.array(instance_mask_predicted_1_1)
            plt.imshow(instance_mask_predicted_1_1_array)
            plt.show()

            #bounding_Boxes_N_M = eval_utils.get_bounding_boxes(instance_mask_matched_N_M).values()
            bounding_Boxes_1_1 = eval_utils.get_bounding_boxes(instance_mask_matched).values()

            #img_boxes_N_M = Image.fromarray(
                #eval_utils.drawBoundingBoxes(np.array(real_img), bounding_Boxes_N_M, (0, 255, 0)).astype(
                   # 'uint8'))
            img_boxes_1_1 = Image.fromarray(
                eval_utils.drawBoundingBoxes(np.array(real_img), bounding_Boxes_1_1, (0, 255, 0)).astype(
                    'uint8'))
            img_boxes_1_1_array = np.array(img_boxes_1_1)
            plt.imshow(img_boxes_1_1_array)
            plt.show()

            #Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU, B_Box_IoU, precision, recall, pixelIoU = eval_utils.get_avg_IoU_AP_AR(
               # instance, instance_mask_matched_N_M)

            Avg_BBox_IoU1_1, AP1_1, AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1 = eval_utils.get_avg_IoU_AP_AR(
                instance, instance_mask_matched)

            #write_results(result_dir, count_naming, Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU,
                         # B_Box_IoU, precision, recall, pixelIoU)

            write_results(join(result_dir, "1_1"), count_naming, Avg_BBox_IoU1_1, AP1_1,
                          AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1)

            """
            write_images(result_dir, count_naming, real_img, semantic_mask_target_img, segmentation_mask_img,
                         instance_mask_target_img, instance_mask_predicted_N_M, img_boxes_N_M)

            write_images(join(result_dir, "1_1"), count_naming, real_img, semantic_mask_target_img,
                         segmentation_mask_img,
                         instance_mask_target_img, instance_mask_predicted_1_1, img_boxes_1_1)

            """
            count_naming += 1


def filter_classes_has_instance(mask):
    image_shape = mask.shape
    has_instance_list = [
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if not np.any(np.all(mask[i, j] == has_instance_list, axis=1)):
                mask[i, j] = [0, 0, 0]

    return mask


def resize_mask(mask, size):
    mask = torch.tensor(mask.astype('float32'))
    if mask.ndim == 3:
        mask = torch.unsqueeze(mask, 0)
    mask = mask.permute((0, 3, 1, 2))

    mask = F.interpolate(input=mask, size=size,
                         mode='bilinear', align_corners=False)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask.numpy()

    plotted_img = Image.fromarray(mask[0].astype(np.uint8))
    # plotted_img.show()
    plotted_array = np.array(plotted_img)

    # Use matplotlib to display the image
    plt.imshow(plotted_array)
    plt.show()

    return mask


def grayscale_to_random_color(grayscale, image_shape, color_list):
    result = np.zeros((image_shape[0], image_shape[1], 3))
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            result[i, j] = color_list[int(grayscale[i, j])]
    return result


if __name__ == "__main__":
    prep_args()
    my_app()
