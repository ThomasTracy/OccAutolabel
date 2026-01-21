import os
import json
import shutil
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

def parse_nuscenes_to_files(nuscenes_root, output_dir, 
                            version='v1.0-mini',
                            parse_num=None):
    """
    解析NuScenes数据集并转存到指定路径
    
    Args:
        nuscenes_root: NuScenes数据集根路径
        output_dir: 输出目录
        version: 数据集版本
    """
    
    # 初始化NuScenes对象
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)
    cam_sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    image_dir = [os.path.join(output_dir, cam) for cam in cam_sensors]
    pointcloud_dir = os.path.join(output_dir, 'lidar')
    pose_dir = os.path.join(output_dir, 'pose')
    calib_dir = os.path.join(output_dir, 'calibration')
    
    for dir_path in image_dir + [pointcloud_dir, pose_dir, calib_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 遍历所有样本
    if parse_num is None:
        parse_num = len(nusc.sample)
    calib_info = {}
    pose_info = {}
    for sample_idx in tqdm(range(parse_num)):
        sample = nusc.sample[sample_idx]
        # print(f"Processing sample {sample_idx + 1}/{len(nusc.sample)}")
        
        # 获取时间戳（使用样本的timestamp）
        timestamp = sample['timestamp']
        
        # 处理相机图像
        for camera_type in cam_sensors:
            
            if camera_type in sample['data']:
                # 获取相机数据
                camera_data = nusc.get('sample_data', sample['data'][camera_type])
                
                # 复制图像文件
                src_image_path = os.path.join(nuscenes_root, camera_data['filename'])
                dst_image_name = f"{timestamp}.jpg"
                dst_image_path = os.path.join(output_dir, camera_type, dst_image_name)
                
                if os.path.exists(src_image_path):
                    shutil.copy2(src_image_path, dst_image_path)
        
        # 处理点云数据
        if 'LIDAR_TOP' in sample['data']:
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            
            # 加载点云
            lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])
            pointcloud = LidarPointCloud.from_file(lidar_path)
            
            # 保存点云为bin文件
            points = pointcloud.points.T  # 转置为 (N, 4)
            pointcloud_name = f"{timestamp}.bin"
            pointcloud_path = os.path.join(pointcloud_dir, pointcloud_name)
            points.astype(np.float32).tofile(pointcloud_path)
        
        # 获取ego_pose信息
        if 'LIDAR_TOP' in sample['data']:
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # 提取位姿信息
            pose_info[timestamp] = {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation']
            }
        
        # 获取标定信息
        cur_calib = {}
        if 'LIDAR_TOP' in sample['data']:
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            
            # 提取标定信息
            cur_calib['lidar'] = {
                'lidar2ego':{
                    'translation': calib_data['translation'],
                    'rotation': calib_data['rotation']
                }
            }
            
            # 添加相机内参
            for camera_type in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                if camera_type in sample['data']:
                    cam_data = nusc.get('sample_data', sample['data'][camera_type])
                    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                    cur_calib[camera_type] = {
                        'cam2ego':{
                            'rotation': cam_calib['rotation'],
                            'translation': cam_calib['translation']
                        },
                        'camera_intrinsic': cam_calib['camera_intrinsic']
                    }
        calib_info[timestamp] = cur_calib

            
    # 保存位姿信息为JSON
    pose_name = f"pose.json"
    pose_path = os.path.join(pose_dir, pose_name)
    with open(pose_path, 'w') as f:
        json.dump(pose_info, f, indent=4)
    
    # 保存标定信息为JSON
    calib_name = f"calibration.json"
    calib_path = os.path.join(calib_dir, calib_name)
    with open(calib_path, 'w') as f:
        json.dump(calib_info, f, indent=4)



def parse_nuscenes_to_clips(nuscenes_root, output_dir, 
                            version='v1.0-mini',
                            parse_num=None):
    """
    解析NuScenes数据集并转存到指定路径，按照场景分段存储到不同clip文件夹中
    
    Args:
        nuscenes_root: NuScenes数据集根路径
        output_dir: 输出目录
        version: 数据集版本
        parse_num: 要解析的场景数量，None表示全部
    """
    
    # 初始化NuScenes对象
    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=True)
    cam_sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 获取要处理的场景列表
    scenes_to_process = nusc.scene
    if parse_num is not None:
        scenes_to_process = scenes_to_process[:parse_num]

    clip_count = 0
    
    # 遍历所有场景
    for scene_idx, scene in enumerate(tqdm(scenes_to_process, desc="Processing scenes")):

        # if clip_count >= 100:
        #     break

        # 创建该场景对应的clip文件夹
        clip_name = f"clip{scene_idx:04d}"
        clip_output_dir = os.path.join(output_dir, clip_name)
        
        # 创建clip目录结构
        camera_dir = os.path.join(clip_output_dir, 'camera')  # 统一的camera文件夹
        image_dirs = [os.path.join(camera_dir, cam) for cam in cam_sensors]  # 相机文件夹在camera下
        pointcloud_dir = os.path.join(clip_output_dir, 'lidar')
        pose_dir = os.path.join(clip_output_dir, 'pose')
        calib_dir = os.path.join(clip_output_dir, 'calibration')
        
        # 创建所有需要的目录
        for dir_path in [camera_dir] + image_dirs + [pointcloud_dir, pose_dir, calib_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 获取该场景的第一个和最后一个sample token
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        
        # 遍历该场景的所有samples
        sample_token = first_sample_token
        sample_count = 0
        
        calib_info = {}
        pose_info = {}
        
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            timestamp = sample['timestamp']
            
            # 处理相机图像
            for camera_type in cam_sensors:
                if camera_type in sample['data']:
                    # 获取相机数据
                    camera_data = nusc.get('sample_data', sample['data'][camera_type])
                    
                    # 复制图像文件到camera下的对应文件夹
                    src_image_path = os.path.join(nuscenes_root, camera_data['filename'])
                    dst_image_name = f"{timestamp}.jpg"
                    dst_image_path = os.path.join(camera_dir, camera_type, dst_image_name)
                    print(src_image_path)
                    print(dst_image_path)
                    
                    if os.path.exists(src_image_path):
                        shutil.copy2(src_image_path, dst_image_path)
            
            # 处理点云数据
            if 'LIDAR_TOP' in sample['data']:
                lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                
                # 加载点云
                lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])
                pointcloud = LidarPointCloud.from_file(lidar_path)
                
                # 保存点云为bin文件
                points = pointcloud.points.T  # 转置为 (N, 4)
                pointcloud_name = f"{timestamp}.bin"
                pointcloud_path = os.path.join(pointcloud_dir, pointcloud_name)
                points.astype(np.float32).tofile(pointcloud_path)
            
            # 获取ego_pose信息
            if 'LIDAR_TOP' in sample['data']:
                lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
                
                # 提取位姿信息
                pose_info[timestamp] = {
                    'translation': ego_pose['translation'],
                    'rotation': ego_pose['rotation']
                }
            
            # 获取标定信息（只需要获取一次，因为同一场景内标定信息相同）
            if sample_count == 0:  # 只在第一个sample时获取标定信息
                cur_calib = {}
                if 'LIDAR_TOP' in sample['data']:
                    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                    
                    # 提取标定信息
                    cur_calib['lidar'] = {
                        'lidar2ego':{
                            'translation': calib_data['translation'],
                            'rotation': calib_data['rotation']
                        }
                    }
                    
                    # 添加相机内参
                    for camera_type in cam_sensors:
                        if camera_type in sample['data']:
                            cam_data = nusc.get('sample_data', sample['data'][camera_type])
                            cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                            cur_calib[camera_type] = {
                                'cam2ego':{
                                    'rotation': cam_calib['rotation'],
                                    'translation': cam_calib['translation']
                                },
                                'camera_intrinsic': cam_calib['camera_intrinsic']
                            }
                calib_info.update(cur_calib)
            
            sample_count += 1
            
            # 移动到下一个sample
            if sample_token == last_sample_token:
                break
            sample_token = sample['next']
        
        # 保存位姿信息为JSON
        pose_name = f"pose.json"
        pose_path = os.path.join(pose_dir, pose_name)
        with open(pose_path, 'w') as f:
            json.dump(pose_info, f, indent=4)
        
        # 保存标定信息为JSON
        calib_name = f"calibration.json"
        calib_path = os.path.join(calib_dir, calib_name)
        with open(calib_path, 'w') as f:
            json.dump(calib_info, f, indent=4)
        
        clip_count += 1
        
        print(f"Finished processing scene {scene_idx}: {scene['name']} with {sample_count} samples")


if __name__ == "__main__":
    # 配置路径
    nuscenes_root = '/data/Data/nuscenes_mini/v1.0-mini'  # 修改为你的NuScenes数据集路径
    output_dir = '/home/robot/data/nusc_mini_full'  # 修改为输出目录
    
    # 解析数据集
    parse_nuscenes_to_clips(nuscenes_root, output_dir, 
                            version='v1.0-mini',
                            parse_num=None)