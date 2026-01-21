import os
import json
import yaml
import torch
import logging
import chamfer
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from copy import deepcopy
import cv2
import pandas as pd


from cores.utils.projector import Projector
from cores.config.config import Config
from collections import Counter, deque
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional

from cores.category import Category


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
        if type(config_path) == Config:
            self.config = config_path
        else:
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
        assert 'lidar' in root_dirs, "lidar dir not found!"
        self.timestamps = self.get_all_timestamps(os.path.join(root, 'lidar'))
        self.lidar_path = os.path.join(root, 'lidar')

        assert 'images' in root_dirs, "images dir not found!"
        self.camera_path = os.path.join(root, 'images')

        assert 'calibration.json' in root_dirs, "calibration.json not found!"
        self.calibrations = self.load_json_data(os.path.join(root, 'calibration.json'))

        assert 'odometry.csv' in root_dirs, "odometry.csv not found!"
        self.poses = self.load_csv_data(os.path.join(root, 'odometry.csv'))
        
        if 'mask' in root_dirs:
            self.mask_path = os.path.join(root, 'mask')
        
        if 'label_map' in self.config:
            with open(self.config['label_map'], 'r') as f:
                self.label_mapping = yaml.safe_load(f)

    def get_all_timestamps(self, data_root):
        """
        获取所有可用的时间戳
        Args:
            data_root: 数据目录字典
        Returns:
            list: 时间戳列表
        """
        # 从文件名中获取时间戳
        files = os.listdir(data_root)
        # timestamps = [f.split('.')[0] for f in lidar_files if f.endswith('.bin')]
        timestamps = [os.path.splitext(file)[0] for file in files]
        
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
    
    def get_calibration(self, sensor_name):
        if 'lidar' in sensor_name.lower():
            return {
                'lidar2ego_rotation': np.array(self.calibrations['lidar']['lidar2ego']['rotation']),
                'lidar2ego_translation': np.array(self.calibrations['lidar']['lidar2ego']['translation'])
            }
        elif 'cam' in sensor_name.lower():
            return {
                'cam2ego_rotation': np.array(self.calibrations[sensor_name]['cam2ego']['rotation']),
                'cam2ego_translation': np.array(self.calibrations[sensor_name]['cam2ego']['translation']),
                'camera_intrinsic': np.array(self.calibrations[sensor_name]['camera_intrinsic']),
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
        prime = np.array([73856093, 19349663, 83492791], dtype=np.int64)
        voxel_ids = np.dot(voxel_indices, prime)

        # 获取唯一体素ID和对应的索引
        _, unique_index, inverse_indices, counts = np.unique(
            voxel_ids, return_inverse=True, return_index=True, return_counts=True
        )
        # 计算照索引获取体素中心点
        voxel_center = (voxel_indices[unique_index] + 0.5) * voxel_size

        # ######################### 计算体素语义 ######################
        num_voxels = len(unique_index)
        num_classes = len(Category)
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
    
    def create_voxel_occupancy_dense(
        self,
        point_cloud: np.ndarray
    ) -> np.ndarray:
        """
        将点云转换为带有语义的体素占据，动态物体具有优先级
        
        Args:
            point_cloud: (N, 4) numpy数组,前3列为xyz坐标,第4列为语义标签
            voxel_size: 体素大小（米）
            min_bound: 体素网格的最小边界 (x_min, y_min, z_min)
            max_bound: 体素网格的最大边界 (x_max, y_max, z_max)
            
        Returns:
            (M, 4) numpy数组,每行为[x, y, z, semantic]
            x, y, z为体素中心坐标,semantic为体素语义
        """
        min_bound = self.config['voxel_range'][:3]
        max_bound = self.config['voxel_range'][3:]
        voxel_size = self.config['voxel_size']

        grid_size = (max_bound - min_bound) / voxel_size
        grid_shape = np.ceil(grid_size).astype(np.int32)  # (length, width, height)
        
        # 3. 计算总网格尺寸并创建所有体素
        length, width, height = grid_shape
        dense_voxel_num = length * width * height
        
        # 4. 生成所有体素的中心坐标
        # 使用向量化方法生成坐标网格
        x_coords = np.arange(length, dtype=np.float32) * voxel_size + min_bound[0] + voxel_size/2
        y_coords = np.arange(width, dtype=np.float32) * voxel_size + min_bound[1] + voxel_size/2
        z_coords = np.arange(height, dtype=np.float32) * voxel_size + min_bound[2] + voxel_size/2
        
        # 创建三维网格并展平
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        dense_voxel_center = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        x = torch.from_numpy(dense_voxel_center).cuda().unsqueeze(0).float()
        y = torch.from_numpy(point_cloud[:,:3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer.forward(x,y)

        # 每个点找其最临近的Voxel赋值
        indices_pc_to_voxel = idx2[0].cpu().numpy()

        dense_semantic = np.ones(dense_voxel_num, dtype=np.int32) * Category.FREE.value

        dense_semantic[indices_pc_to_voxel] = point_cloud[:, 3]

        dense_voxels_with_semantic = np.hstack([dense_voxel_center, dense_semantic[:, np.newaxis]])

        return dense_voxels_with_semantic