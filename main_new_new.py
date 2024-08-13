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
import torch
import torch.nn.functional as F


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
    count = [0,1,2,3,4,5,6,7,8,9,10,11,1,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    #count = list(range(101))
    preds_formatted = []
    targets_formatted = []

    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    for i, batch in enumerate(tqdm(loader)):
        if i not in count:
           continue

        with torch.no_grad():
            img = batch["img"].cuda()
            semantic_target = batch["label"].cuda()



            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()



            cluster_crf_numpy = dense_crf(img.detach().cpu()[0], cluster_probs.detach().cpu()[0])
            cluster_crf = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in cluster_crf_numpy], dim=0)
            cluster_crf = cluster_crf.unsqueeze(0)
            cluster_preds = cluster_crf.argmax(1).cuda()


            linear_crf_numpy = dense_crf(img.detach().cpu()[0], linear_probs.detach().cpu()[0])
            linear_crf = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in linear_crf_numpy], dim=0)
            linear_crf = linear_crf.unsqueeze(0)
            linear_preds = linear_crf.argmax(1).cuda()


            model.test_linear_metrics.update(linear_preds, semantic_target)
            model.test_cluster_metrics.update(cluster_preds, semantic_target)




            tb_metrics = {
                **model.test_linear_metrics.compute(),
                **model.test_cluster_metrics.compute(), }




    for i, batch in enumerate(tqdm(loader)):
        if i not in count:
            continue

        with torch.no_grad():
            img = batch["img"].cuda()
            depth = batch["depth"]
            depth = torch.squeeze(depth).numpy()
            instance_target = batch["instance"]
            rgb_img = batch["real_img"]
            rgb_image = Image.fromarray(rgb_img[0].squeeze().numpy().astype(np.uint8))
            rgb_image_array = np.array(rgb_image)
            plt.imshow(rgb_image_array)
            plt.show()


            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)


            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()



            cluster_crf_numpy = dense_crf(img.detach().cpu()[0], cluster_probs.detach().cpu()[0])
            cluster_crf = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in cluster_crf_numpy], dim=0)
            cluster_crf = cluster_crf.unsqueeze(0)
            cluster_preds = cluster_crf.argmax(1)






            predictions = model.test_cluster_metrics.map_clusters(cluster_preds)
            predictions = predictions.squeeze(0)
            predictions = predictions.cpu().numpy()
            predictions_img = Image.fromarray(predictions.astype(np.uint8))



            if cfg.clustering_algorithm == "dbscan" or cfg.clustering_algorithm == "bgmm":
                predicted_instance_mask = maskD.segmentation_to_instance_mask(predictions_img, depth,
                                                                              image_shape,
                                                                              clustering_algorithm=cfg.clustering_algorithm,
                                                                              epsilon=cfg.epsilon,
                                                                              min_samples=cfg.min_samples,
                                                                              project_data=True)
            elif cfg.clustering_algorithm == "geo":
                masks = maskD.get_segmentation_masks(predictions_img)

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


                for k in return_dict.keys():
                    labels = len(np.unique(return_dict[k])) - 1
                    class_instance_mask = return_dict[k]

                    class_instance_mask = np.where(class_instance_mask != 0,
                                                   class_instance_mask + current_num_instances, 0)
                    current_num_instances += labels

                    predicted_instance_mask = np.add(predicted_instance_mask, class_instance_mask)

            else:
                raise ValueError("Clustering algorithm not supported. Please choose dbscan, bgmm or geo.")


            plt.imshow(predicted_instance_mask)
            plt.show()
            predictions_tensor = torch.tensor(predicted_instance_mask)



            targets = torch.tensor(instance_target).squeeze(0)
            plt.imshow(targets.numpy())
            plt.show()

            targets = targets.long()
            predictions_tensor = predictions_tensor.long()
            pred_one_hot = F.one_hot(predictions_tensor).permute(2, 0, 1).to(torch.uint8)
            target_one_hot = F.one_hot(targets).permute(2, 0, 1).to(torch.uint8)

            pred_labels = torch.arange(pred_one_hot.shape[0])
            target_labels = torch.arange(target_one_hot.shape[0])

            pred_scores = torch.ones(pred_one_hot.shape[0])

            pred_dict = {
                "labels": pred_labels,
                "masks": pred_one_hot,  # Binary masks for each predicted instance
                "scores": pred_scores,  # Dummy scores
            }
            preds_formatted.append(pred_dict)

            target_dict = {
                "labels": target_labels,
                "masks": target_one_hot,  # Binary masks for each ground truth instance
            }
            targets_formatted.append(target_dict)





    metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    metric.update(preds_formatted, targets_formatted)
    result = metric.compute()
    print(result)





def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask

    return_dict[procnum] = labelRangeImage(rel_depth)




if __name__ == "__main__":
    prep_args()
    my_app()


