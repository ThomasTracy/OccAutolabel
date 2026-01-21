from cores.config.config import Config
import numpy as np
import yaml
import os
import sys

from cores.autolabel import OCCAutolabelwithLidarSegmentation, OCCAutolabelwithUEPose
from tools.bin2pcd import bin_to_pcd
from tools.lidar_on_cam_video import main as lidar_on_cam_video
from vis_voxel_to_video import main as vis_voxel_to_video


if __name__ == '__main__':

    print("--------------------------------------- run autolabel pipeline ---------------------------------------")
    config_path = "/home/jszr/Code/Autolabel/occ_autolabel/configs/config.yaml"
    autolabel = OCCAutolabelwithUEPose(config_path)
    autolabel.process_multi_frame()

    print("--------------------------------------- convert full pcd from bin to pcd ---------------------------------------")
    root_path = autolabel.config['data_root']
    bin_path = os.path.join(root_path, "world_points.bin")
    pcd_path = os.path.join(root_path, "world_points.pcd")
    if os.path.exists(bin_path):
        bin_to_pcd(bin_path, pcd_path, dimension=4)

    print("--------------------------------------- generate gt voxel image ---------------------------------------")
    annotation_root = os.path.join(root_path, "annotations")
    vis_voxel_to_video(annotation_root)
    
    print("--------------------------------------- generate lidar cam projection video ---------------------------------------")
    lidar_on_cam_video(root_path)