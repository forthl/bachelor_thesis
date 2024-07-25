from multiprocessing import Pool

import numpy as np  # dont remove this otherwise gets stuck in infinite loop
import cv2
from datasets.depth_dataset import ContrastiveDepthDataset
from utils.eval_segmentation import batched_crf
from utils.eval_segmentation import dense_crf
from utils.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import utils.maskDepth_new as maskD
from utils.train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
import torchvision.transforms as T
import utils.evaluation_utils_new as eval_utils
import utils.cluster_algorithms_new
import open3d as o3d

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_directory_path = cfg.results_dir
    result_dir = join(result_directory_path, "results/predictions/Semantic_mask__1/")
    os.makedirs(join(result_dir, "Metrics"), exist_ok=True)
    os.makedirs(join(result_dir, "real_img"), exist_ok=True)
    os.makedirs(join(result_dir, "segmentation_target"), exist_ok=True)
    os.makedirs(join(result_dir, "segmentation_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "instance_target"), exist_ok=True)
    os.makedirs(join(result_dir, "instance_predicted"), exist_ok=True)
    os.makedirs(join(result_dir, "bounding_boxes"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

    color_list = random_colors

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

    count_naming = 0
    count = 0

    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    for i, batch in enumerate(tqdm(loader)):
        if count <= -1:
            count += 1
            continue

        with (torch.no_grad()):
            img = batch["img"].cuda()
            label = batch["label"].cuda()
            depth = batch["depth"]
            real_img = batch["real_img"]
            instance = batch["instance"]

            transToImg = T.ToPILImage()
            real_image = transToImg(real_img[0])
            real_img = real_img[0]
            instance = instance[0].numpy()
            instance = eval_utils.normalize_labels(instance)
            depth_img = transToImg(depth[0])
            depth = torch.squeeze(depth)
            depth = depth.numpy()

            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            # linear_crf = torch.from_numpy(dense_crf(img.detach().cpu()[0], linear_probs.detach().cpu()[0])).cuda()
            cluster_crf_numpy = dense_crf(img.detach().cpu()[0], cluster_probs.detach().cpu()[0])
            cluster_crf = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in cluster_crf_numpy], dim=0)
            cluster_crf = cluster_crf.unsqueeze(0)
            cluster_crf = cluster_crf.argmax(1).cuda()

            model.test_cluster_metrics.update(cluster_crf, label)
            tb_metrics = {
                **model.test_linear_metrics.compute(),
                **model.test_cluster_metrics.compute(), }

            segmentation_mask_colored = model.label_cmap[
                model.test_cluster_metrics.map_clusters(cluster_crf.cpu())].astype(np.uint8)
            segmentation_mask = model.label_cmap[label[0].cpu()].astype(np.uint8)
            segmentation_mask_img = Image.fromarray(segmentation_mask_colored[0])

            filtered_segmentation_mask = filter_classes_has_instance(
                segmentation_mask_colored[0])  # sets backgroung to 0
            filtered_segmentation_mask_img = Image.fromarray(filtered_segmentation_mask.astype(np.uint8))



            predicted_instance_mask = np.sum(filtered_segmentation_mask,axis=2)

            predicted_instance_mask = eval_utils.normalize_labels(predicted_instance_mask)
            instance = eval_utils.normalize_labels(instance)

            instance_mask_not_matched = np.zeros(image_shape)

            predicted_instance_ids = np.unique(predicted_instance_mask)

            assignments = eval_utils.get_assigment(predicted_instance_mask,
                                                   instance)  # targetIDs, matched InstanceIDs

            num_matched_instances = assignments[0].size

            not_matched_instance_ids = np.setdiff1d(predicted_instance_ids, assignments[1])

            instance_mask_matched = np.zeros(image_shape)

            for i, val in enumerate(assignments[1]):  # this is correct (assignments[1])
                mask = np.where(predicted_instance_mask == val, assignments[0][i], 0)
                instance_mask_matched = instance_mask_matched + mask

            for i, id in enumerate(not_matched_instance_ids):
                instance_mask_not_matched = np.add(instance_mask_not_matched,
                                                   np.where(predicted_instance_mask == id,
                                                            num_matched_instances + i, 0))

            instance_mask = Image.fromarray(
                grayscale_to_random_color(instance, image_shape, color_list).astype(np.uint8))
            # instance_mask.show()

            if cfg.eval_N_M:
                instance_mask_matched_N_M = np.add(instance_mask_matched, instance_mask_not_matched)

            instance_mask_predicted = Image.fromarray(
                grayscale_to_random_color(instance_mask_matched_N_M, image_shape, color_list).astype(np.uint8))
            # instance_mask_predicted.show()

            Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU, B_Box_IoU, precision, recall, pixelIoU = eval_utils.get_avg_IoU_AP_AR(
                instance, instance_mask_matched_N_M)

            Avg_BBox_IoU1_1, AP1_1, AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1 = eval_utils.get_avg_IoU_AP_AR(
                instance, instance_mask_matched)

            boundingBoxes = eval_utils.get_bounding_boxes(instance_mask_matched).values()

            img_boxes = eval_utils.drawBoundingBoxes(real_img.numpy(), boundingBoxes, (0, 255, 0))
            img_boxes = Image.fromarray(img_boxes.astype('uint8'))
            #img_boxes.show()

            write_results(result_dir, count_naming, Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU, B_Box_IoU, precision, recall,
                          pixelIoU)
            write_results(join(result_dir, "1_1"), count_naming, Avg_BBox_IoU1_1, AP1_1,
                                 AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1)

            write_images(result_dir, count_naming, real_image, segmentation_mask_img, segmentation_mask_img,
                         instance_mask,
                         instance_mask_predicted, img_boxes)

            count_naming += 1

            print("breakpoint")


if __name__ == "__main__":
    prep_args()
    my_app()