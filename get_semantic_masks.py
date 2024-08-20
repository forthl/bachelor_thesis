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
    semantic_path = join(cfg.result_dir, "semantic_masks")
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
    for i, batch in enumerate(tqdm(loader)):


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

    predictions_list = []
    predicted_semantic_mask_colored_list = []


    for i, batch in enumerate(tqdm(loader)):


        with torch.no_grad():
            img = batch["img"].cuda()



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
            predictions = predictions.cpu()
            predictions = predictions.squeeze()

            predicted_semantic_mask_colored = model.label_cmap[
                model.test_cluster_metrics.map_clusters(cluster_preds.cpu())].astype(np.uint8)

            predictions_list.append(predictions)
            predicted_semantic_mask_colored_list.append(predicted_semantic_mask_colored)

    os.makedirs(semantic_path, exist_ok=True)

    # Save the predictions list to a file
    predictions_path = join(semantic_path,  "predictions.txt")
    with open(predictions_path, "w") as f:
        for index, pred in enumerate(predictions_list):
            f.write(f"Entry {index}:\n")
            f.write(f"Predictions:\n{pred.tolist()}\n")  # Convert numpy array to list for readable output
            f.write("\n")

    # Save the predicted semantic masks to a separate file
    predicted_masks_path = join(semantic_path, "predicted_semantic_masks_colored.txt")
    with open(predicted_masks_path, "w") as f:
        for index, mask in enumerate(predicted_semantic_mask_colored_list):
            f.write(f"Entry {index}:\n")
            f.write(f"Predicted Semantic Mask Colored:\n{mask.tolist()}\n")
            f.write("\n")

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved predicted semantic masks to {predicted_masks_path}")



if __name__ == "__main__":
    prep_args()
    my_app()