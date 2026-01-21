import os
import sys
import json
import time
import yaml
import torch
import logging
import chamfer
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import open3d as o3d
from copy import deepcopy
from PIL import Image
import cv2
import pandas as pd

import random

from cores.utils.projector import Projector
from cores.config.config import Config


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=True,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud

def poisson_rebuild(pcd, depth, n_threads=8, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

class OCCAutolabelBase:
    def __init__(self, config_path):
        self.config = Config(config_path)
        self.inti_data(self.config['data_root'])
        self.projector = Projector()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def inti_data(self, root):
        root_dirs = os.listdir(root)
        self.poses = []
        self.calibrations = {}
        self.timestamps = []
        self.label_mapping = {}
        self.lidar_path = ''
        self.camera_path = ''
        self.mask_path = ''
        root_dirs = os.listdir(root)
        if 'lidar' in root_dirs:
            self.timestamps = self.get_all_timestamps(os.path.join(root, 'lidar'))
            self.lidar_path = os.path.join(root, 'lidar')
        if 'images' in root_dirs:
            self.camera_path = os.path.join(root, 'images')
        if 'mask' in root_dirs:
            self.mask_path = os.path.join(root, 'mask')
        if 'pose' in root_dirs:
            self.poses = self.load_json_data(os.path.join(root, 'pose','pose.json'))
        if 'calibration.json' in root_dirs:
            self.calibrations = self.load_json_data(os.path.join(root, 'calibration.json'))
        if 'label_map' in self.config:
            with open(self.config['label_map'], 'r') as f:
                self.label_mapping = yaml.safe_load(f)
        if 'odometry.csv' in root_dirs:
            self.poses = self.load_csv_data(os.path.join(root, 'odometry.csv'))

    def get_all_timestamps(self, data_root):
        """
        获取所有可用的时间戳
        Args:
            data_root: 数据目录字典
        Returns:
            list: 时间戳列表
        """
        # 从LiDAR目录获取时间戳
        lidar_files = os.listdir(data_root)
        timestamps = [f.split('.')[0] for f in lidar_files if f.endswith('.bin')]
        
        # 按时间戳排序
        timestamps.sort(key=lambda x: float(x))
        
        return timestamps

    def load_json_data(self, json_path):
        """
        加载标定数据
        Args:
            json_path: 标定文件路径
        Returns:
            dict: 包含所有相机和LiDAR标定信息的字典
        """
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        return json_data
    
    def load_csv_data(self, odom_path):
        """
        加载里程计数据
        假设CSV格式包含: timestamp, x, y, z, qx, qy, qz, qw
        """
        odom_df = pd.read_csv(odom_path)
        
        # 检查必要的列是否存在
        required_columns = ['timestamp', 'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
        for col in required_columns:
            if col not in odom_df.columns:
                raise ValueError(f"里程计文件缺少必要的列: {col}")
            
        result = {}
        for _, row in odom_df.iterrows():
            timestamp = str(row['timestamp'])
            result[timestamp] = {
                'translation': [
                    float(row['position_x']),
                    float(row['position_y']),
                    float(row['position_z'])
                ],
                'rotation': [
                    float(row['orientation_w']),
                    float(row['orientation_x']),
                    float(row['orientation_y']),
                    float(row['orientation_z'])
                ]
            }
        
        return result

    def get_pose(self, timestamp):
        """
        加载指定时间戳的位姿数据
        Args:
            pose_path: 位姿文件路径
            timestamp: 时间戳
        Returns:
            dict: 位姿信息
        """
        
        # 查找最接近的时间戳
        timestamps = list(self.poses.keys())
        closest_timestamp = min(timestamps, key=lambda x: abs(float(x) - float(timestamp)))
        return self.poses[closest_timestamp]
    
    def get_calibration(self, timestamp, sensor_name):
        if 'lidar' in sensor_name.lower():
            return {
                'lidar2ego_rotation': np.array(self.calibrations[timestamp]['lidar']['lidar2ego']['rotation']),
                'lidar2ego_translation': np.array(self.calibrations[timestamp]['lidar']['lidar2ego']['translation'])
            }
        elif 'cam' in sensor_name.lower():
            return {
                'cam2ego_rotation': np.array(self.calibrations[timestamp][sensor_name]['cam2ego']['rotation']),
                'cam2ego_translation': np.array(self.calibrations[timestamp][sensor_name]['cam2ego']['translation']),
                'camera_intrinsic': np.array(self.calibrations[timestamp][sensor_name]['camera_intrinsic']),
            }
        else:
            return {}

    def load_lidar_pointcloud(self, timestamp, point_dim=3):
        """
        加载LiDAR点云数据
        Args:
            lidar_path: LiDAR数据目录路径
            timestamp: 时间戳
        Returns:
            np.array: 点云数据 (N, 4) [x, y, z, intensity]
        """
        lidar_file = os.path.join(self.lidar_path, f"{timestamp}.bin")
        if not os.path.exists(lidar_file):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_file}")
        
        # 读取二进制点云文件
        pointcloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, point_dim)
        return pointcloud

    def load_camera_image(self, timestamp, camera_name):
        """
        加载相机图像
        Args:
            camera_path: 相机数据目录路径
            timestamp: 时间戳
            camera_name: 相机名称
        Returns:
            np.array: 图像数据
        """
        image_file = os.path.join(self.camera_path, camera_name, f"{timestamp}.jpg")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")
        
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def create_voxel_occupancy_with_rebuild(self, point_cloud):
        """
        创建体素Occupancy
        Args:
            point_cloud: 带有语义的点云数据
        Returns:
            np.array: 体素Occupancy
        """
        if point_cloud.shape[1] < 4:
            self.logger.error("Point cloud does not contain semantic! shape: {}", point_cloud.shape)
            return None
        pc_range = self.config['pc_range']
        mask = (point_cloud[:, 0] > pc_range[0]) & (point_cloud[:, 0] < pc_range[3]) \
                & (point_cloud[:, 1] > pc_range[1]) & (point_cloud[:, 1] < pc_range[4]) \
                & (point_cloud[:, 2] > pc_range[2]) & (point_cloud[:, 2] < pc_range[5])
        point_cloud = point_cloud[mask]

        pc_o3d = o3d.geometry.PointCloud()
        pc_with_normal = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        with_normal = preprocess_cloud(pc_o3d, self.config['max_nn'],normals=True)
        pc_with_normal.points = with_normal.points
        pc_with_normal.normals = with_normal.normals

        ################## poisson重建  ##############
        mesh, _ = poisson_rebuild(pc_with_normal, 
                                  self.config['depth'], 
                                  self.config['n_threads'],
                                  self.config['min_density'])
        scene_points = np.asarray(mesh.vertices, dtype=float)
        mask = (scene_points[:, 0] > pc_range[0]) & (scene_points[:, 0] < pc_range[3]) \
                & (scene_points[:, 1] > pc_range[1]) & (scene_points[:, 1] < pc_range[4]) \
                & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
        scene_points = scene_points[mask]

        # Voxelization
        pcd_np = scene_points
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / self.config['voxel_size']
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / self.config['voxel_size']
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / self.config['voxel_size']
        pcd_np = np.floor(pcd_np).astype(np.int_)
        occ_size = [
            int((pc_range[3] - pc_range[0]) / self.config['voxel_size']),
            int((pc_range[4] - pc_range[1]) / self.config['voxel_size']),
            int((pc_range[5] - pc_range[2]) / self.config['voxel_size'])
        ]
        voxel = np.zeros(occ_size)
        voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

        ################## voxel 坐标转换为Lidar坐标系  ##############
        gt_ = voxel
        x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
        y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
        z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        vv = np.stack([X, Y, Z], axis=-1)
        fov_voxels = vv[gt_ > 0]
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * self.config['voxel_size']
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        ################## 最临近法将点云语义映射到体素网格中 ##############
        dense_voxels = fov_voxels

        x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
        y = torch.from_numpy(point_cloud[:,:3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer.forward(x,y)
        indices = idx1[0].cpu().numpy()
        # d1, d2, idx1, idx2 = chamfer_distance(dense_voxels,sparse_voxels_semantic[:,:3])
        # indices = idx1

        dense_semantic = point_cloud[:, 3][np.array(indices)]
        for key, value in self.label_mapping['label2label'].items():
            mask = dense_semantic == key
            dense_semantic[mask] = value
        dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

        # to voxel coordinate
        pcd_np = dense_voxels_with_semantic
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / self.config['voxel_size']
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / self.config['voxel_size']
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / self.config['voxel_size']
        dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int_)

        return dense_voxels_with_semantic
    

    def create_voxel_occupancy_slim(self, point_cloud):
        '''
        直接对点云进行体素化，得到每个体素的中心点
        体素的语义为每个体素中最多的点的语义
        '''
        pc_xyz = point_cloud[:, :3]
        pc_semantic = point_cloud[:, 3]
        voxel_size = self.config['voxel_size']
        valid_voxel_point_num = 5

        # ######################### 创建体素并计算center ######################
        voxel_indices = np.floor(pc_xyz / voxel_size).astype(np.int64)
        # 生成唯一体素ID
        prime =  primes = np.array([73856093, 19349663, 83492791], dtype=np.int64)
        voxel_ids = np.dot(voxel_indices, prime)

        # 获取唯一体素ID和对应的索引
        _, unique_index, inverse_indices, counts = np.unique(
            voxel_ids, return_inverse=True, return_index=True, return_counts=True
        )
        # 计算照索引获取体素中心点
        voxel_center = (voxel_indices[unique_index] + 0.5) * voxel_size

        # ######################### 计算体素语义 ######################
        num_voxels = len(unique_index)
        num_classes = len(self.label_mapping['label2label'])
        hist_2d = np.zeros((num_voxels, num_classes), dtype=np.int32)
        linear_indices = (inverse_indices * num_classes + pc_semantic).astype(np.int64)

        np.add.at(hist_2d.ravel(), linear_indices, 1)
        
        # 7. 找到每个体素中数量最多的语义标签
        voxel_semantics = np.argmax(hist_2d, axis=1)
        res = np.concatenate([voxel_center, voxel_semantics[:, np.newaxis]], axis=1)

        # save_path = "/home/jszr/Data/OCC_Autolabel/DATA/TOWN/clip1/annotations.bin"
        # with open(save_path,'wb') as f:
        #     res.astype(np.float32).tofile(f)
        # input("===+++++===")

        return res



class OCCAutolabelwithMask(OCCAutolabelBase):
    def __init__(self, config):
        super().__init__(config)

    def load_semantic_mask(self, mask_path, timestamp, camera_name):
        """
        加载语义分割mask
        Args:
            mask_path: mask数据目录路径
            timestamp: 时间戳
            camera_name: 相机名称
        Returns:
            np.array: 语义mask
        """
        mask_file = os.path.join(mask_path, camera_name, f"{timestamp}.png")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        return mask

    def assign_semantics_from_masks(self, points, points_img, mask, image_shape):
        """
        根据图像分割mask为点云分配语义标签
        Args:
            points: 点云数据
            points_img: 投影后的图像坐标
            mask: 语义分割mask
            label_mapping: 标签映射字典
            image_shape: 图像形状
        Returns:
            np.array: 语义标签
        """
        semantics = np.zeros((points.shape[0], 1), dtype=np.uint8)
        
        if len(points_img) == 0:
            return semantics
        
        for i, img_coord in enumerate(points_img):
            x, y = int(round(img_coord[0])), int(round(img_coord[1]))
            
            # 检查点是否在图像范围内
            if 0 <= x < image_shape[0] and 0 <= y < image_shape[1]:
                # 从分割mask获取语义标签
                mask_label = mask[y, x]
                
                # 映射到学习标签
                if mask_label in self.label_mapping['label2label']:
                    semantics[i] = self.label_mapping['label2label'][mask_label]
        
        return semantics

    
    def process_single_frame(self, timestamp, lidar_points=None):
        """
        处理单个时间戳的数据帧
        Args:
            timestamp: 时间戳
            data_root: 数据目录字典
            calibration_data: 标定数据
            label_mapping: 标签映射
            config: 配置参数
        Returns:
            dict: 处理结果
        """
        
        # 加载LiDAR点云
        if lidar_points is None:
            lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])
        
        # 初始化语义点云
        semantic_points = np.zeros((lidar_points.shape[0], 4))
        semantic_points[:, :3] = lidar_points[:, :3]  # 坐标
        semantic_points[:, 3] = 0  # 默认语义标签
        
        # 处理每个相机
        camera_names = self.config['camera_types']
        
        for cam_name in camera_names:
            # 检查相机数据是否存在
            cam_dir = os.path.join(self.camera_path, cam_name)
            if not os.path.exists(cam_dir):
                continue
            
            # 加载图像和mask
            image = self.load_camera_image(timestamp, cam_name)
            mask = self.load_semantic_mask(self.mask_path, timestamp, cam_name)
            
            cam_calib = self.get_calibration(timestamp,cam_name)
            camera_matrix = cam_calib.get('camera_intrinsic', None)
            distortion_coeffs = cam_calib.get('distortion_coeffs', None)
            camera_to_ego_rotation = cam_calib.get('cam2ego_rotation', None)
            camera_to_ego_translation = cam_calib.get('cam2ego_translation', None)

            # 因为之前在多帧点云拼接时已经做过lidar到ego的转换，所以这里直接设为None
            lidar_to_ego_rotation = None
            lidar_to_ego_translation = None
            # lidar_calib = self.get_calibration(timestamp,'lidar')
            # lidar_to_ego_rotation = lidar_calib.get('lidar2ego_rotation', None)
            # lidar_to_ego_translation = lidar_calib.get('lidar2ego_translation', None)

            # cv2.imread 获取的图片w和h顺序反向
            image_shape = np.zeros(2)
            image_shape[0], image_shape[1] = image.shape[1], image.shape[0]
            points2d, valid_mask = self.projector.projection(
                lidar_points,
                image_shape=image_shape,
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs,
                camera_to_ego_rotation=camera_to_ego_rotation,
                camera_to_ego_translation=camera_to_ego_translation,
                lidar_to_ego_rotation=lidar_to_ego_rotation,
                lidar_to_ego_translation=lidar_to_ego_translation
            )

            # 分配语义标签
            cam_semantics = self.assign_semantics_from_masks(
                lidar_points[valid_mask, :3], 
                points2d, 
                mask, 
                image_shape
            )
            
            # 更新语义点云
            semantic_points[valid_mask, 3] = np.maximum(
                semantic_points[valid_mask, 3], cam_semantics.flatten()
            )
            res = self.create_voxel_occupancy(semantic_points)
            save_path = os.path.join(self.config['save_path'],"{}.npy".format(timestamp))
            np.save(save_path, res)
        self.logger.info("------------ finish processing frame {} ------------".format(timestamp))

        return semantic_points
    
    def process_multi_frame(self, debug_timestamp=None):
        for i, timestamp in enumerate(self.timestamps):
            if debug_timestamp is not None and timestamp != debug_timestamp:
                continue
            self.logger.info("------------ start processing frame {} ------------".format(timestamp))
            frame_num = self.config['point_multi_frame_num']
            selected_poses = []
            selected_points = []
            ref_pose = self.get_pose(timestamp)
            for j in range(frame_num):
                if (i-j < 0):
                    break
                selected_tp = self.timestamps[i-j]
                selected_poses.append(self.poses[selected_tp])
                lidar_points = self.load_lidar_pointcloud(selected_tp,self.config['point_dim'])[:,:3]

                # 将点云从lidar坐标系转换到ego坐标系
                calib = self.get_calibration(selected_tp,'lidar')
                lidar_to_ego = self.projector.to_matrix4x4(calib['lidar2ego_rotation'],
                                                        calib['lidar2ego_translation'])
                lidar_points_homo = self.projector.to_homo_coord(lidar_points)
                points_on_ego = lidar_points_homo @ lidar_to_ego.T
                # 去除自车点云
                mask = (points_on_ego[:, 0] > 3) | (points_on_ego[:, 0] < -1) | \
                        (points_on_ego[:, 1] > 1) | (points_on_ego[:, 1] < -1)
                points_on_ego = points_on_ego[mask]

                selected_points.append(points_on_ego[:,:3])
            
            selected_points = self.projector.multi_lidar_concat(selected_points, selected_poses, ref_pose)
            # np.save("full_points.npy", selected_points)
            # input("Press Enter to continue...")
            self.process_single_frame(timestamp, selected_points)
            if debug_timestamp is not None:
                return
        return


class PointColorGenerator(OCCAutolabelBase):
    '''
    通过点云投影图片方式为点云分配颜色
    '''
    def __init__(self, config):
        super().__init__(config)

    def assign_color_from_image(self, points, points_img, image, image_shape):
        """
        根据图像为点云分配rgb颜色
        Args:
            points: 三维点云数据
            points_img: 投影后的图像坐标
            image: 图片数据
            label_mapping: 标签映射字典
            image_shape: 图像形状
        Returns:
            np.array: 语义标签
        """
        semantics = np.zeros((points.shape[0], 3), dtype=np.uint8)
        
        if len(points_img) == 0:
            return semantics
        
        for i, img_coord in enumerate(points_img):
            x, y = int(round(img_coord[0])), int(round(img_coord[1]))
            
            # 检查点是否在图像范围内
            if 0 <= x < image_shape[0] and 0 <= y < image_shape[1]:
                # 从image获取颜色标签
                semantics[i] = image[y, x]
        
        return semantics
    
    def process_single_frame(self, timestamp, lidar_points=None):
        """
        处理单个时间戳的数据帧
        Args:
            timestamp: 时间戳
            lidar_points: 若
            calibration_data: 标定数据
            label_mapping: 标签映射
            config: 配置参数
        Returns:
            dict: 处理结果
        """
        
        # 加载LiDAR点云
        if lidar_points is None:
            lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])
        
        # 初始化语义点云
        rgb_points = np.zeros((lidar_points.shape[0], 6))
        rgb_points[:, :3] = lidar_points[:, :3]  # 坐标
        rgb_points[:, 3:] = 0  # rgb颜色
        
        # 处理每个相机
        camera_names = self.config['camera_types']
        
        for cam_name in camera_names:
            # 检查相机数据是否存在
            cam_dir = os.path.join(self.camera_path, cam_name)
            if not os.path.exists(cam_dir):
                continue
            
            # 加载图像和mask
            image = self.load_camera_image(timestamp, cam_name)
            
            cam_calib = self.get_calibration(timestamp,cam_name)
            camera_matrix = cam_calib.get('camera_intrinsic', None)
            distortion_coeffs = cam_calib.get('distortion_coeffs', None)
            camera_to_ego_rotation = cam_calib.get('cam2ego_rotation', None)
            camera_to_ego_translation = cam_calib.get('cam2ego_translation', None)

            # 因为之前在多帧点云拼接时已经做过lidar到ego的转换，所以这里直接设为None
            lidar_to_ego_rotation = None
            lidar_to_ego_translation = None
            # lidar_calib = self.get_calibration(timestamp,'lidar')
            # lidar_to_ego_rotation = lidar_calib.get('lidar2ego_rotation', None)
            # lidar_to_ego_translation = lidar_calib.get('lidar2ego_translation', None)

            # cv2.imread 获取的图片w和h顺序反向
            image_shape = np.zeros(2)
            image_shape[0], image_shape[1] = image.shape[1], image.shape[0]
            points2d, valid_mask = self.projector.projection(
                lidar_points,
                image_shape=image_shape,
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs,
                camera_to_ego_rotation=camera_to_ego_rotation,
                camera_to_ego_translation=camera_to_ego_translation,
                lidar_to_ego_rotation=lidar_to_ego_rotation,
                lidar_to_ego_translation=lidar_to_ego_translation
            )

            # 分配语义标签
            cam_rgb = self.assign_color_from_image(
                lidar_points[valid_mask, :3], 
                points2d, 
                image, 
                image_shape
            )
            
            # 更新语义点云
            rgb_points[valid_mask, 3:] = cam_rgb
            save_path = os.path.join(self.config['save_path'],"{}.npy".format(timestamp))
            np.save(save_path, rgb_points)
        self.logger.info("------------ finish processing frame {} ------------".format(timestamp))

        return rgb_points
    
    def process_multi_frame(self, debug_timestamp=None):
        for i, timestamp in enumerate(self.timestamps):
            if debug_timestamp is not None and timestamp != debug_timestamp:
                continue
            self.logger.info("------------ start processing frame {} ------------".format(timestamp))
            frame_num = self.config['point_multi_frame_num']
            selected_poses = []
            selected_points = []
            ref_pose = self.get_pose(timestamp)
            for j in range(frame_num):
                if (i-j < 0):
                    break
                selected_tp = self.timestamps[i-j]
                selected_poses.append(self.poses[selected_tp])
                lidar_points = self.load_lidar_pointcloud(selected_tp,self.config['point_dim'])[:,:3]

                # 将点云从lidar坐标系转换到ego坐标系
                calib = self.get_calibration(selected_tp,'lidar')
                lidar_to_ego = self.projector.to_matrix4x4(calib['lidar2ego_rotation'],
                                                        calib['lidar2ego_translation'])
                lidar_points_homo = self.projector.to_homo_coord(lidar_points)
                points_on_ego = lidar_points_homo @ lidar_to_ego.T
                # 去除自车点云
                mask = (points_on_ego[:, 0] > 3) | (points_on_ego[:, 0] < -1) | \
                        (points_on_ego[:, 1] > 1) | (points_on_ego[:, 1] < -1)
                points_on_ego = points_on_ego[mask]

                selected_points.append(points_on_ego[:,:3])
            
            selected_points = self.projector.multi_lidar_concat(selected_points, selected_poses, ref_pose)
            # np.save("full_points.npy", selected_points)
            # input("Press Enter to continue...")
            self.process_single_frame(timestamp, selected_points)
            if debug_timestamp is not None:
                return

        return
    

class OCCAutolabelwithLidarSegmentation(OCCAutolabelBase):
    def __init__(self, config):
        super(OCCAutolabelwithLidarSegmentation, self).__init__(config)
        self.total_points = []
        self.total_poses = []
        # self.slam_poses = self.init_poses()

    def init_poses(self):
        pose_path = os.path.join(self.config['data_root'], 'poses.txt')
        transmatrix = {}
        with open(pose_path, 'r') as f:
            poses = f.readlines()
            for pose in poses:
                pose = pose.strip().split()
                timestamp = pose[1][:-3]
                trans_matrix = np.array(pose[2:]).reshape(4, 4).astype(np.float32)
                transmatrix[timestamp] = trans_matrix
        return transmatrix

    def process_single_frame(self, timestamp, points=None):
         # 加载LiDAR点云
        if points is not None:
             lidar_points = points
        else:
            lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])

        res = self.create_voxel_occupancy_slim(lidar_points)
        save_path = os.path.join(self.config['save_path'],"{}.npz".format(timestamp))
        res = {'occ_gt': res}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **res)

        return lidar_points
    
    def multi_lidar_concat(self):
        '''
        将多个激光点云拼接成单个点云
        拼接到第一帧中
        '''
        points_in_world = []
        
        for i, points in enumerate(self.total_points):
            if len(points) == 0:
                continue
            
            # 获取当前帧的位姿
            current_pose = self.total_poses[i]
            current_position = np.array(current_pose['translation'])
            current_quaternion = np.array(current_pose['rotation'])
            # current_rotation = R.from_quat(current_quaternion).as_matrix()
            cur_to_world = self.to_matrix4x4(current_quaternion, current_position)
            
            points_homo = self.to_homo_coord(points)
            # transform_matrix = np.linalg.inv(self.pose0_to_world_transform) @ cur_to_world
            p_in_world = (points_homo @ cur_to_world.T)[:, :3]
            # 添加到合并点云
            points_in_world.append(p_in_world)

        points_in_world = np.vstack(points_in_world)

        return points_in_world
    
    def get_calibration(self, sensor_name):
        if 'lidar' in sensor_name.lower():
            return {
                'ego2lidar_rotation': np.array(self.calibrations['lidar']['rotation']),
                'ego2lidar_translation': np.array(self.calibrations['lidar']['translation'])
            }
        elif 'cam' in sensor_name.lower():
            return {
                'ego2cam_rotation': np.array(self.calibrations[sensor_name]['rotation']),
                'ego2cam_translation': np.array(self.calibrations[sensor_name]['translation']),
                'camera_intrinsic': np.array(self.calibrations[sensor_name]['camera_intrinsic']),
            }
        else:
            return {}
    
    def trans_angular_to_matrix4x4(self, translation, angular):
        rx, ry, rz = angular
        Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
        
        R = Rz @ Ry @ Rx

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = np.array(translation)
        return transform

    def process_multi_frame(self, debug_timestamp=None):
        for i, timestamp in enumerate(self.timestamps):
            if debug_timestamp is not None and timestamp != debug_timestamp:
                continue
            frame_num = self.config['point_multi_frame_num']
            frame_radius = self.config['multi_frame_radius']
            selected_points = []
            selected_semantics = []
            full_points_in_world = []
            full_semantics = []
            ref_pose = self.get_pose(timestamp)
            ref_position = np.array(ref_pose['translation'])
            ref_quaternion = np.array(ref_pose['rotation'])
            ref_to_world = self.projector.to_matrix4x4(ref_quaternion, ref_position)
            
            ref_lidar_points = self.load_lidar_pointcloud(timestamp,self.config['point_dim'])
            ref_pos = ref_lidar_points[0, :3]
            ref_angular = ref_lidar_points[1,:3]
            ref_to_world = self.trans_angular_to_matrix4x4(ref_pos, ref_angular)

            # 以当前帧为中心，向前后取 frame_radius 数量的帧进行拼接
            # 前面取不满的就后面多取，保证一共满足 frame_radius * 2 + 1 帧
            selected_tp = []
            if 2 * frame_radius + 1 >= len(self.timestamps):
                selected_tp = self.timestamps
            start_idx = max(0, i - frame_radius)
            end_idx = min(len(self.timestamps), i + frame_radius + 1)
            if start_idx == 0:
                end_idx = 2 * frame_radius + 1
            if end_idx == len(self.timestamps):
                start_idx = len(self.timestamps) - 2 * frame_radius - 1
            selected_tp = self.timestamps[start_idx:end_idx]
            # selected_tp = random.sample(self.timestamps, 100)
            # 取当前帧的前 frame_num 帧进行拼接
            for tp in selected_tp:
                cur_pose = self.get_pose(tp)
                current_position = np.array(cur_pose['translation'])
                current_quaternion = np.array(cur_pose['rotation'])
                cur_to_world = self.projector.to_matrix4x4(current_quaternion, current_position)

                lidar_points = self.load_lidar_pointcloud(tp,self.config['point_dim'])

                cur_pos = lidar_points[0, :3]
                cur_angular = ref_lidar_points[1,:3]
                cur_to_world = self.trans_angular_to_matrix4x4(cur_pos, cur_angular)

                lidar_points = lidar_points[2:]
                lidar_points_xyz = lidar_points[:,:3]
                lidar_points_semantic = lidar_points[:,-1]

                # map intensity to semantic label
                intensity2label = self.label_mapping['label2label']
                for k,v in intensity2label.items():
                    lidar_points_semantic[lidar_points_semantic == k] = v

                # 将点云从lidar坐标系转换到ego坐标系
                calib = self.get_calibration('lidar')
                ego_to_lidar = self.projector.to_matrix4x4(calib['ego2lidar_rotation'],
                                                        calib['ego2lidar_translation'])
                lidar_to_ego = np.linalg.inv(ego_to_lidar)
                lidar_points_homo = self.projector.to_homo_coord(lidar_points_xyz)
                points_on_ego = lidar_points_homo @ lidar_to_ego.T

                # 去除本机点云
                mask1 = (points_on_ego[:, 0] > 0.5) | (points_on_ego[:, 0] < -0.5) | \
                        (points_on_ego[:, 1] > 0.5) | (points_on_ego[:, 1] < -0.5)

                # ego --> world --> reference frame
                points_in_world = points_on_ego @ cur_to_world.T

                # 高度范围在世界坐标系中截取
                mask2 = (points_in_world[:, 2] > self.config['pc_range'][2]) & (points_in_world[:, 2] < self.config['pc_range'][5])

                points_on_ref = points_in_world @ np.linalg.inv(ref_to_world).T

                # 只保留pc_range范围内的点
                mask3 = (points_on_ref[:, 0] > self.config['pc_range'][0]) & (points_on_ref[:, 0] < self.config['pc_range'][3]) & \
                        (points_on_ref[:, 1] > self.config['pc_range'][1]) & (points_on_ref[:, 1] < self.config['pc_range'][4])
                
                mask = mask1 & mask2 & mask3
                # mask = mask1

                # selected_points.append(points_in_world[:,:3])
                full_points_in_world.append(points_in_world[:,:3])
                full_semantics.extend(lidar_points_semantic)
                selected_points.append(points_on_ref[mask][:,:3])
                selected_semantics.extend(lidar_points_semantic[mask])
            
            selected_points = np.vstack(selected_points)
            selected_semantics = np.array(selected_semantics)
            selected_points = np.concatenate([selected_points, selected_semantics[:, None]], axis=1)

            full_points_in_world = np.vstack(full_points_in_world)
            full_semantics = np.array(full_semantics)
            full_points_in_world = np.concatenate([full_points_in_world, full_semantics[:, None]], axis=1)

            # save_path = os.path.join(self.config['data_root'], 'multi_lidar', '{}.bin'.format(timestamp))
            save_path = os.path.join(self.config['data_root'], 'world_points.bin')
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    full_points_in_world.astype(np.float32).tofile(f)
            # with open(save_path, 'wb') as f:
            #     selected_points.astype(np.float32).tofile(f)
            # input("Press Enter to continue ~~~~~~~")

            self.process_single_frame(timestamp, selected_points)
            self.logger.info("------------ finish processing frame {} ------------".format(timestamp))
            if debug_timestamp is not None:
                return
