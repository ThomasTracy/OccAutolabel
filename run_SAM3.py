from cores.config.config import Config
import numpy as np
import yaml
import os
import sys

from cores.autolabel import OCCAutolabelwithMask
from tools.bin2pcd import bin_to_pcd
from tools.lidar_on_cam_video import main as lidar_on_cam_video
from vis.vis_voxel_to_video import main as vis_voxel_to_video


if __name__ == '__main__':

    print("============================== run autolabel pipeline ==============================")
    config_path = "/home/robot/Code/Autolabel/occ_autolabel/configs/config_sam3_nuscenes.yaml"
    config = Config(config_path)
    
    root_path = config['data_root']
    save_path = config['save_path']
    clips = os.listdir(root_path)
    for clip in clips:
        print("---------------------------- clip: {} ----------------------------".format(clip))
        config['data_root'] = os.path.join(root_path, clip)
        config['save_path'] = os.path.join(root_path, clip, save_path)

        autolabel = OCCAutolabelwithMask(config)
        autolabel.process_multi_frame()

    # print("--------------------------------------- convert full pcd from bin to pcd ---------------------------------------")
    # root_path = autolabel.config['data_root']
    # bin_path = os.path.join(root_path, "world_points.bin")
    # pcd_path = os.path.join(root_path, "world_points.pcd")
    # if os.path.exists(bin_path):
    #     bin_to_pcd(bin_path, pcd_path, dimension=4)

    # print("--------------------------------------- generate gt voxel image ---------------------------------------")
    # annotation_root = os.path.join(root_path, "annotations")
    # vis_voxel_to_video(annotation_root)
    
    # print("--------------------------------------- generate lidar cam projection video ---------------------------------------")
    # lidar_on_cam_video(root_path)