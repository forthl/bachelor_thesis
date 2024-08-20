import ast
import json

from datasets.depth_dataset import ContrastiveDepthDataset
from utils.drive_seg_geo_transformation import labelRangeImage
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
import utils.evaluation_utils as eval_utils
from multiprocessing import Manager, Process
import utils.unsupervised_metrics
import utils.utils_folder.unsupervised_metrics_new
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torch.nn.functional as F


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_directory_path = join(cfg.results_dir, cfg.clustering_algorithm)
    semantic_path = join(cfg.results_dir, "semantic_masks")
    os.makedirs(join(result_directory_path, "Metrics"), exist_ok=True)
    os.makedirs(join(result_directory_path, "real_img"), exist_ok=True)
    os.makedirs(join(result_directory_path, "semantic_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "semantic_predicted"), exist_ok=True)
    os.makedirs(join(result_directory_path, "instance_target"), exist_ok=True)
    os.makedirs(join(result_directory_path, "instance_predicted"), exist_ok=True)

    considering_background = cfg.considering_background






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


    preds_formatted = []
    targets_formatted = []
    metric_per_image_list = []

    real_images =[]
    predicted_semantic_images = []
    predicted_instance_images = []
    target_instance_images = []
    target_semantic_images = []

    predictions_path = os.path.join(semantic_path, "predictions.txt")
    predicted_masks_path = os.path.join(semantic_path, "predicted_semantic_masks_colored.txt")

    # Load predictions from file
    predictions_list = []
    with open(predictions_path, "r") as f:
        lines = f.readlines()
        for i in range(2, len(lines), 4):
            pred_line = lines[i].strip()
            pred_array = np.array(eval(pred_line))
            predictions_list.append(torch.tensor(pred_array, dtype=torch.long))

    # Load predicted semantic masks from file
    predicted_semantic_mask_colored_list = []
    with open(predicted_masks_path, "r") as f:
        lines = f.readlines()
        for i in range(2, len(lines), 4):
            mask_line = lines[i].strip()
            mask_array = np.array(eval(mask_line), dtype=np.uint8)
            predicted_semantic_mask_colored_list.append(mask_array)


    ten_constant_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    #count = [222]


    for i, batch in enumerate(tqdm(loader)):
        #if i not in count:
            #continue

        with torch.no_grad():

            depth = batch["depth"]
            depth = torch.squeeze(depth).numpy()
            instance_target = batch["instance"]

            semantic_target = batch["label"].cuda()

            rgb_img = batch["real_img"]
            rgb_image = Image.fromarray(rgb_img[0].squeeze().numpy().astype(np.uint8))
            real_images.append(rgb_image)
            rgb_image_array = np.array(rgb_image)
            plt.imshow(rgb_image_array)
            plt.show()

            label_cpu = semantic_target.cpu()
            semantic_mask_target_img = Image.fromarray(model.label_cmap[label_cpu[0].squeeze()].astype(np.uint8))
            target_semantic_images.append(semantic_mask_target_img)
            semantic_mask_target_img_array = np.array(semantic_mask_target_img)
            plt.imshow(semantic_mask_target_img_array)
            plt.show()

            # Load predictions and predicted masks from the lists loaded from files
            predictions = predictions_list[i]
            predicted_semantic_mask_colored = predicted_semantic_mask_colored_list[i]

            # Show the predicted semantic mask
            predicted_semantic_mask_img = Image.fromarray(predicted_semantic_mask_colored[0])
            predicted_semantic_images.append(predicted_semantic_mask_img)
            predicted_semantic_mask_img_array = np.array(predicted_semantic_mask_img)
            plt.imshow(predicted_semantic_mask_img_array)
            plt.show()


            #filtering the instances the model shpuld predict (e.g. only cars, persons, etc.) like in cityscapes specified
            #other instances are set to 0 (background)

            filtered_semantic_mask = filter_classes_has_instance(predicted_semantic_mask_colored[0])
            filtered_semantic_mask_img = Image.fromarray(filtered_semantic_mask.astype(np.uint8))
            filtered_semantic_mask_img_array = np.array(filtered_semantic_mask_img)
            plt.imshow(filtered_semantic_mask_img_array)
            plt.show()



            if cfg.clustering_algorithm == "dbscan" or cfg.clustering_algorithm == "bgmm":

                predicted_instance_mask, labels_list, instance_list = maskD.segmentation_to_instance_mask(filtered_semantic_mask_img, depth,
                                                                              image_shape,
                                                                              clustering_algorithm=cfg.clustering_algorithm,
                                                                              epsilon=cfg.epsilon,
                                                                              min_samples=cfg.min_samples,
                                                                              project_data=True)
            elif cfg.clustering_algorithm == "geo":
                masks, classes = maskD.get_segmentation_masks(filtered_semantic_mask_img)
                labels_list = []

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

                class_index = 0


                for k in return_dict.keys():
                    labels = len(np.unique(return_dict[k])) - 1
                    class_instance_mask = return_dict[k]

                    class_instance_mask = np.where(class_instance_mask != 0,
                                                   class_instance_mask + current_num_instances, 0)
                    current_num_instances += labels

                    predicted_instance_mask = np.add(predicted_instance_mask, class_instance_mask)

                    for i in range(class_index, class_index + labels):
                        labels_list.append(classes[k])
                    class_index += labels

                instance_list = list(range(0, current_num_instances))

            elif cfg.clustering_algorithm == "no_clustering":

                masks, classes = maskD.get_segmentation_masks(filtered_semantic_mask_img)
                masks.pop(0)
                classes.pop(0)
                instance_list = []

                labels_list = []
                predicted_instance_mask = np.zeros(masks[0].shape)
                class_index = 0
                for mask in masks:
                    value = classes[class_index]
                    predicted_instance_mask[mask] = value
                    instance_list.append(value)
                    labels_list.append(classes[class_index])
                    class_index += 1


            else:
                raise ValueError("Clustering algorithm not supported. Please choose dbscan, bgmm, geo or no_clustering.")




            if considering_background:
                labels_list.insert(0,0)


            pred_labels = torch.tensor(labels_list)

            targets = torch.tensor(instance_target).squeeze(0)

            plt.imshow(targets.numpy())
            plt.show()

            plt.imshow(predicted_instance_mask)
            plt.show()

            instance_target_normalized = eval_utils.normalize_labels(instance_target[0].numpy())
            instance_mask_target_img = Image.fromarray(
                grayscale_to_random_color(instance_target_normalized, image_shape, color_list).astype(np.uint8))
            target_instance_images.append(instance_mask_target_img)
            instance_mask_target_img_array = np.array(instance_mask_target_img)
            plt.imshow(instance_mask_target_img_array)
            plt.show()




            predictions_tensor = torch.tensor(predicted_instance_mask)

            predictions_tensor_normalized = eval_utils.normalize_labels(predictions_tensor.numpy())

            instance_mask_prediction_img = Image.fromarray(
                grayscale_to_random_color(predictions_tensor_normalized, image_shape, color_list).astype(np.uint8))
            predicted_instance_images.append(instance_mask_prediction_img)
            instance_mask_prediction_img_array = np.array(instance_mask_prediction_img)
            plt.imshow(instance_mask_prediction_img_array)
            plt.show()


            #unique_pred_labels = torch.unique(predictions_tensor)
            pred_masks = torch.zeros((len(instance_list), image_shape[0], image_shape[1]), dtype=torch.uint8)
            index = 0
            for i in instance_list:
                pred_masks[index] = (predictions_tensor == i).to(torch.uint8)
                index += 1

            unique_target_labels = targets.unique()
            num_labels = len(unique_target_labels)
            target_masks = torch.zeros((num_labels, image_shape[0], image_shape[1]), dtype=torch.uint8)
            for i, target in enumerate(unique_target_labels):
                target_masks[i] = (targets == target).to(torch.uint8)






            target_labels = torch.zeros(target_masks.shape[0], dtype=torch.uint8)
            for i in range(0, target_masks.shape[0]):
                binary_array = target_masks[i].cpu().numpy()
                class_array = semantic_target.squeeze().cpu().numpy() * binary_array
                selected_values = class_array[binary_array == 1]
                selected_values_tensor = torch.tensor(selected_values, dtype=torch.float32)
                if selected_values_tensor.numel() == 0:
                    most_frequent_value = 0

                else:
                    most_frequent_value = selected_values_tensor.mode().values.item()
                if torch.isnan(torch.tensor(most_frequent_value)):
                    most_frequent_value = 0

                target_labels[i] = int(most_frequent_value)

            if target_labels.size(0) == 1: #in this case there is no instance in the image that should be detected (only background)
                if considering_background:
                    filtered_target_labels = target_labels
                    filtered_target_masks = target_masks
                else:
                    filtered_target_labels = torch.tensor([])
                    filtered_target_masks = torch.tensor([])

            else:
                filtered_target_labels, filtered_target_masks = filter_ignore_in_eval(target_labels, target_masks, considering_background)



            pred_scores = torch.ones(pred_masks.shape[0])
            #shape = pred_one_hot.shape[0]
            #pred_scores = torch.full((shape,), 0.5)

            pred_dict = {
                "labels": pred_labels,
                "masks": pred_masks,  # Binary masks for each predicted instance
                "scores": pred_scores,  # Dummy scores
            }
            preds_formatted.append(pred_dict)

            target_dict = {
                "labels": filtered_target_labels,
                "masks": filtered_target_masks,  # Binary masks for each ground truth instance
            }
            targets_formatted.append(target_dict)



            metric_per_image = MeanAveragePrecision(iou_type="segm", class_metrics=True)
            metric_per_image.update([pred_dict], [target_dict])
            result = metric_per_image.compute()
            result_serializable = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            map_value = result_serializable.get("map")
            metric_per_image_list.append(map_value)





    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    metric.update(preds_formatted, targets_formatted)
    result = metric.compute()
    result_serializable = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    metrics_dir = join(result_directory_path, "Metrics")
    results_file = join(metrics_dir, "mAP_results.txt")
    with open(results_file, 'w') as f:
        for key, value in result_serializable.items():
            f.write(f"{key}: {value}\n")
    print(result)

    good_images_dir = join(result_directory_path, "good_images")
    bad_images_dir = join(result_directory_path, "bad_images")
    constant_images_dir = join(result_directory_path, "constant_images")
    os.makedirs(good_images_dir, exist_ok=True)
    os.makedirs(bad_images_dir, exist_ok=True)
    os.makedirs(constant_images_dir, exist_ok=True)

    best_images_indices, worst_images_indices = get_best_and_worst_images(metric_per_image_list, top_n=10)

    for i in best_images_indices:
        real_images[i].save(join(good_images_dir, f"real_img_{i}.png"))
        target_semantic_images[i].save(join(good_images_dir, f"semantic_target_{i}.png"))
        predicted_semantic_images[i].save(join(good_images_dir, f"semantic_predicted_{i}.png"))
        target_instance_images[i].save(join(good_images_dir, f"instance_target_{i}.png"))
        predicted_instance_images[i].save(join(good_images_dir, f"instance_predicted_{i}.png"))

    for i in worst_images_indices:
        real_images[i].save(join(bad_images_dir, f"real_img_{i}.png"))
        target_semantic_images[i].save(join(bad_images_dir, f"semantic_target_{i}.png"))
        predicted_semantic_images[i].save(join(bad_images_dir, f"semantic_predicted_{i}.png"))
        target_instance_images[i].save(join(bad_images_dir, f"instance_target_{i}.png"))
        predicted_instance_images[i].save(join(bad_images_dir, f"instance_predicted_{i}.png"))

    for i in ten_constant_indices:
        real_images[i].save(join(constant_images_dir, f"real_img_{i}.png"))
        target_semantic_images[i].save(join(constant_images_dir, f"semantic_target_{i}.png"))
        predicted_semantic_images[i].save(join(constant_images_dir, f"semantic_predicted_{i}.png"))
        target_instance_images[i].save(join(constant_images_dir, f"instance_target_{i}.png"))
        predicted_instance_images[i].save(join(constant_images_dir, f"instance_predicted_{i}.png"))



def get_best_and_worst_images(metric_per_image, top_n=10):


    sorted_indices = sorted(range(len(metric_per_image)), key=lambda i: metric_per_image[i], reverse=True)

    # Indizes der 10 besten Bilder
    best_images_indices = sorted_indices[:top_n]

    # Indizes der 10 schlechtesten Bilder
    worst_images_indices = sorted_indices[-top_n:]

    return best_images_indices, worst_images_indices



def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask

    return_dict[procnum] = labelRangeImage(rel_depth)


def filter_ignore_in_eval(target_labels, target_masks, considering_background=False):
    if considering_background:
        consider_in_eval = [0, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    else:
        consider_in_eval = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    filtered_target_labels = []
    filtered_target_masks = []
    for i, label in enumerate(target_labels):
        if label in consider_in_eval:
            filtered_target_labels.append(label)
            filtered_target_masks.append(target_masks[i])

    return torch.tensor(filtered_target_labels), torch.stack(filtered_target_masks)






if __name__ == "__main__":
    prep_args()
    my_app()