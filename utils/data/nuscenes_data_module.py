import torch

from torchvision import transforms
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes

from utils.utils_folder.representation import pointcloudOnImage, plotGraph, plotGraphs
from utils.utils_folder.data_utils import generatePointCloudFromRangeImage, generateQuadTreeFromRangeImage, generateRangeImageFromPointCloud, generateRangeImageFromTree
from utils.drive_seg_geo_transformation import removeGround, labelRangeImage, find_NNs

class nuScenes(Dataset):
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        samples = [self.getSamplesForScene(
            scene['first_sample_token'], scene['last_sample_token']) for scene in self.nusc.scene]
        self.samples = [s for sample in samples for s in sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        # cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT_RIGHT'])
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pointcloud, colors, image = self.nusc.explorer.map_pointcloud_to_image(
            lidar_data['token'], cam_front_data['token']
        )

        transform = transforms.Compose([transforms.PILToTensor()])

        image = transform(image).permute(0, 2, 1).T
        pointcloud = torch.tensor(pointcloud)
        pointcloud[2, :] = torch.tensor(colors)

        return pointcloud, image, self.geoPointSegs(pointcloud, image)

    def collate_fn(data):
        min = 2**32
        images = []
        depths = []
        batchsize = len(data)

        # To discuss: currently only cutting away, should this be randomized?

        for (d, i) in data:
            min = d.shape[1] if d.shape[1] < min else min

        indices = range(0, min)
        for (d, i) in data:
            images.append(i)
            depths.append(d[:, indices])

        imgs = torch.tensor(
            (batchsize, images[0].shape[0], images[0].shape[1]), dtype=float)
        torch.cat(images, out=imgs)

        dps = torch.tensor(
            (batchsize, depths[0].shape[0], depths[0].shape[1]), dtype=float)
        torch.cat(depths, out=dps)

        return (imgs, dps)

    def getSamplesForScene(self, first_sample_token, last_sample_token, samples=[]):
        if first_sample_token == last_sample_token:
            return samples
        sample = self.nusc.get('sample', first_sample_token)
        samples.append(sample)
        self.getSamplesForScene(sample['next'], last_sample_token, samples)
        return samples

    def geoPointSegs(self, pointcloud, image):
        pointcloud = torch.round(pointcloud.T).long()
        range_image = generateRangeImageFromPointCloud(pointcloud, image.T.shape[1:])
        pointcloudOnImage(generatePointCloudFromRangeImage(range_image), image, 'Initial PointCloud')
        range_without_ground, mask = removeGround(range_image)
        pointcloudOnImage(generatePointCloudFromRangeImage(range_without_ground), image, 'Ground Removal')

        segments = labelRangeImage(range_without_ground)
        
        pointcloudOnImage(generatePointCloudFromRangeImage(segments), image, 'Segmentation')
        
        interpolation = find_NNs(segments, mask)

        plotGraph(interpolation, 'densify & pad')


