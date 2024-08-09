from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from torch import dtype, Tensor

from datasets.depth_dataset import ContrastiveDepthDataset
from utils.drive_seg_geo_transformation import labelRangeImage
from utils.eval_segmentation import dense_crf
from utils.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import utils.maskDepth as maskD
from utils.train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import utils.evaluation_utils as eval_utils
from multiprocessing import Manager, Process
import utils.unsupervised_metrics
import utils.utils_folder.unsupervised_metrics_new
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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

    UnsupervisedMetrics = utils.utils_folder.unsupervised_metrics_new.UnsupervisedMetrics(cfg.num_classes, cfg.num_predictions)

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
    count = [0,1]





    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    for i, batch in enumerate(tqdm(loader)):
        if i not in count:
           continue

        with torch.no_grad():
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

            if cfg.clustering_algorithm == "dbscan" or cfg.clustering_algorithm == "bgmm":
                predicted_instance_mask = maskD.segmentation_to_instance_mask(filtered_semantic_mask_img, depth,
                                                                              image_shape,
                                                                              clustering_algorithm=cfg.clustering_algorithm,
                                                                              epsilon=cfg.epsilon,
                                                                              min_samples=cfg.min_samples,
                                                                              project_data=True)
            elif cfg.clustering_algorithm == "geo":
                masks = maskD.get_segmentation_masks(filtered_semantic_mask_img)
                # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road buildingi)
                masks.pop(0)

                manager = Manager()
                return_dict = manager.dict()
                jobs = []
                for i in range(len(masks)):
                    p = Process(target=worker, args=(
                        i, return_dict, depth, masks[i]))
                    jobs.append(p)
                    p.start()

                for proc in jobs:
                    proc.join()

                current_num_instances = 0
                predicted_instance_mask = np.zeros(masks[0].shape)
                # fig, axeslist = plt.subplots(ncols=3, nrows=3)

                for k in return_dict.keys():
                    labels = len(np.unique(return_dict[k])) - 1
                    class_instance_mask = return_dict[k]

                    class_instance_mask = np.where(class_instance_mask != 0,
                                                   class_instance_mask + current_num_instances, 0)
                    current_num_instances += labels

                    predicted_instance_mask = np.add(predicted_instance_mask, class_instance_mask)

            else:
                raise ValueError("Clustering algorithm not supported. Please choose dbscan, bgmm or geo.")

            segment_features = extract_segment_features(predicted_instance_mask, code)

            predictions = torch.tensor(predicted_instance_mask)
            targets = torch.tensor(instance_target)


            UnsupervisedMetrics.update(segment_features, predictions, targets)









    predicted_masks, targets = UnsupervisedMetrics.compute()





    preds_formatted = []
    targets_formatted = []


    for pred, target in zip(predicted_masks, targets):

        pred_dict = {
            "labels": pred.flatten(),  # labels (Tensor): integer tensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
            "masks": torch.nn.functional.one_hot(pred),  # Convert to boolean masks
        }
        preds_formatted.append(pred_dict)

        target_dict = {
            "labels": target.flatten(),  # Flattened labels
            "masks": torch.nn.functional.one_hot(target),  # Convert to boolean masks
        }
        targets_formatted.append(target_dict)

    # TODO: predictions need to be transformed to the torchmetrics format,
    #  in particular for instance segmentation (for this you need the original
    #  predictions)
    # update the instance segmentation metrics from torchmetrics
    # See https://cocodataset.org/#detection-eval and the corresponding paper
    # on how to evaluate instance segmentation

    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    metric.update(preds_formatted, targets_formatted)
    result = metric.compute()
    print(result)





def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask

    return_dict[procnum] = labelRangeImage(rel_depth)


def extract_segment_features(instance_mask, feature_map):
    if isinstance(instance_mask, np.ndarray):
        instance_mask = torch.from_numpy(instance_mask).to(feature_map.device)

    instance_mask = instance_mask.cpu()
    unique_segments = np.unique(instance_mask)
    num_segments = len(unique_segments)
    num_features = feature_map.shape[1]
    height, width = feature_map.shape[2], feature_map.shape[3]

    segment_features = torch.zeros((num_segments, num_features), device=feature_map.device)

    for i, segment_id in enumerate(unique_segments):
        segment_mask = (instance_mask == segment_id)
        segment_mask = segment_mask.cpu().numpy()
        segment_mask = torch.from_numpy(segment_mask).unsqueeze(0).unsqueeze(0).to(feature_map.device)  # (1, H, W)
        segment_mask = torch.nn.functional.interpolate(segment_mask.float(), size =(height,width), mode='nearest').bool()

        # Extract features for the segment
        segment_mask = segment_mask.squeeze(0)
        segment_features[i] = torch.mean(feature_map * segment_mask, dim=(2,3)).view(num_features)

    return segment_features




if __name__ == "__main__":
    prep_args()
    my_app()


