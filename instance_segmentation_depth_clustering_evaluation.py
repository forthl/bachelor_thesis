import matplotlib.pyplot as plt

from datasets.depth_dataset import ContrastiveDepthDataset
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
import utils.evaluation_utils_new as eval_utils


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_directory_path = join(cfg.results_dir, cfg.clustering_algorithm)
    os.makedirs(join(result_directory_path, "1_1", "Metrics"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "real_img"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "semantic_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "semantic_predicted"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "instance_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "instance_predicted"), exist_ok=True)
    os.makedirs(join(result_directory_path, "1_1", "bounding_boxes"), exist_ok=True)
    os.makedirs(join(result_directory_path, "Metrics"), exist_ok=True)
    os.makedirs(join(result_directory_path, "real_img"), exist_ok=True)
    os.makedirs(join(result_directory_path, "semantic_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "semantic_predicted"), exist_ok=True)
    os.makedirs(join(result_directory_path, "instance_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "instance_predicted"), exist_ok=True)
    os.makedirs(join(result_directory_path, "bounding_boxes"), exist_ok=True)

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
    count = [26]

    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    for i, batch in enumerate(tqdm(loader)):
        if i not in count:
            continue

        with (torch.no_grad()):
            img = batch["img"].cuda()
            semantic_target = batch["label"].cuda()
            depth = batch["depth"]
            rgb_img = batch["real_img"]
            instance_target = batch["instance"]

            instance_target = eval_utils.normalize_labels(instance_target[0].numpy())
            depth = torch.squeeze(depth).numpy()

            rgb_image = Image.fromarray(rgb_img[0].squeeze().numpy().astype(np.uint8))
            rgb_image_array = np.array(rgb_image)
            plt.imshow(rgb_image_array)
            plt.show()
            label_cpu = semantic_target.cpu()

            semantic_mask_target_img = Image.fromarray(model.label_cmap[label_cpu[0].squeeze()].astype(np.uint8))
            semantic_mask_target_img_array = np.array(semantic_mask_target_img)
            plt.imshow(semantic_mask_target_img_array)
            plt.show()
            instance_mask_target_img = Image.fromarray(
                grayscale_to_random_color(instance_target, image_shape, color_list).astype(np.uint8))
            instance_mask_target_img_array = np.array(instance_mask_target_img)
            plt.imshow(instance_mask_target_img_array)
            plt.show()

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

            model.test_cluster_metrics.update(cluster_crf, semantic_target)
            tb_metrics = {
                **model.test_linear_metrics.compute(),
                **model.test_cluster_metrics.compute(), }

            predicted_semantic_mask_colored = model.label_cmap[
                model.test_cluster_metrics.map_clusters(cluster_crf.cpu())].astype(np.uint8)
            predicted_semantic_mask_img = Image.fromarray(predicted_semantic_mask_colored[0])
            predicted_semantic_mask_img_array = np.array(predicted_semantic_mask_img)
            plt.imshow(predicted_semantic_mask_img_array)
            plt.show()

            filtered_semantic_mask = filter_classes_has_instance(
                predicted_semantic_mask_colored[0])  # sets backgroung to 0
            filtered_semantic_mask_img = Image.fromarray(filtered_semantic_mask.astype(np.uint8))
            filtered_semantic_mask_img_array = np.array(filtered_semantic_mask_img)
            plt.imshow(filtered_semantic_mask_img_array)
            plt.show()

            if cfg.resize_to_original:
                filtered_semantic_mask = resize_mask(filtered_semantic_mask, image_shape)
                filtered_semantic_mask_img = Image.fromarray(filtered_semantic_mask[0].astype(np.uint8))

            predicted_instance_mask = maskD.segmentation_to_instance_mask(filtered_semantic_mask_img, depth,
                                                                          image_shape, clustering_algorithm=cfg.clustering_algorithm,
                                                                          epsilon=cfg.epsilon, min_samples=cfg.min_samples,
                                                                          project_data=True)

            predicted_instance_mask = eval_utils.normalize_labels(predicted_instance_mask)
            instance_target = eval_utils.normalize_labels(instance_target)
            assignments = eval_utils.get_assigment(predicted_instance_mask,
                                                   instance_target)  # targetIDs, matched InstanceIDs

            instance_mask_matched = np.zeros(image_shape)
            instance_mask_not_matched = np.zeros(image_shape)

            predicted_instance_ids = np.unique(predicted_instance_mask)

            num_matched_instances = assignments[0].size

            not_matched_instance_ids = np.setdiff1d(predicted_instance_ids, assignments[1])

            for i, val in enumerate(assignments[1]):  # this is correct (assignments[1])
                mask = np.where(predicted_instance_mask == val, assignments[0][i], 0)
                instance_mask_matched = instance_mask_matched + mask

            for i, id in enumerate(not_matched_instance_ids):
                instance_mask_not_matched = np.add(instance_mask_not_matched,
                                                   np.where(predicted_instance_mask == id, num_matched_instances + i,
                                                            0))


            ############## N_M evaluation #########################

            instance_mask_matched_N_M = np.add(instance_mask_matched, instance_mask_not_matched)

            instance_mask_predicted_N_M = Image.fromarray(
                grayscale_to_random_color(instance_mask_matched_N_M, image_shape, color_list).astype(np.uint8))
            instance_mask_predicted_N_M_array = np.array(instance_mask_predicted_N_M)
            plt.imshow(instance_mask_predicted_N_M_array)
            plt.show()

            bounding_Boxes_N_M = eval_utils.get_bounding_boxes(instance_mask_matched_N_M).values()

            img_boxes_N_M = Image.fromarray(
                eval_utils.drawBoundingBoxes(np.array(rgb_image), bounding_Boxes_N_M, (0, 255, 0)).astype('uint8'))
            img_boxes_N_M_array = np.array(img_boxes_N_M)
            plt.imshow(img_boxes_N_M_array)
            plt.show()

            Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU, B_Box_IoU, precision, recall, pixelIoU = eval_utils.get_avg_IoU_AP_AR(
                instance_target, instance_mask_matched_N_M)

            write_results(result_directory_path, count_naming, Avg_BBox_IoU, AP, AR, Avg_Pixel_IoU,
                          B_Box_IoU, precision, recall, pixelIoU)

            write_images(result_directory_path, count_naming, rgb_image, semantic_mask_target_img, predicted_semantic_mask_img,
                         instance_mask_target_img, instance_mask_predicted_N_M, img_boxes_N_M)

            ################# 1_1 evaluation  ##################

            instance_mask_predicted_1_1 = Image.fromarray(
                grayscale_to_random_color(instance_mask_matched, image_shape, color_list).astype(np.uint8))
            instance_mask_predicted_1_1_array = np.array(instance_mask_predicted_1_1)
            plt.imshow(instance_mask_predicted_1_1_array)
            plt.show()

            bounding_Boxes_1_1 = eval_utils.get_bounding_boxes(instance_mask_matched).values()

            img_boxes_1_1 = Image.fromarray(
                eval_utils.drawBoundingBoxes(np.array(rgb_image), bounding_Boxes_1_1, (0, 255, 0)).astype('uint8'))
            img_boxes_1_1_array = np.array(img_boxes_1_1)
            plt.imshow(img_boxes_1_1_array)
            plt.show()

            Avg_BBox_IoU1_1, AP1_1, AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1 = eval_utils.get_avg_IoU_AP_AR(
                instance_target, instance_mask_matched)

            write_results(join(result_directory_path, "1_1"), count_naming, Avg_BBox_IoU1_1, AP1_1,
                          AR1_1, Avg_Pixel_IoU1_1, B_Box_IoU1_1, precision1_1, recall1_1, pixelIoU1_1)

            write_images(join(result_directory_path, "1_1"), count_naming, rgb_image, semantic_mask_target_img,
                         predicted_semantic_mask_img,
                         instance_mask_target_img, instance_mask_predicted_1_1, img_boxes_1_1)

            count_naming += 1

            print("breakpoint")


if __name__ == "__main__":
    prep_args()
    my_app()