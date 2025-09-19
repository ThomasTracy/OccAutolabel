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
from scipy.spatial.transform import Rotation
from scipy import ndimage
import open3d as o3d
from copy import deepcopy
from PIL import Image
import cv2

from cores.utils.projector import Projector


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


class OCCAutolabel:
    def __init__(self, data_dirs, config):
        self.data_dirs = data_dirs
        self.config = config
        self.poses = self.load_json_data(data_dirs['pose'])
        self.calibrations = self.load_json_data(data_dirs['calibration'])
        self.timestamps = self.get_all_timestamps(data_dirs)
        with open(data_dirs['label_mapping'], 'r') as f:
            self.label_mapping = yaml.safe_load(f)
        self.projector = Projector()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def get_all_timestamps(self, data_dirs):
        """
        获取所有可用的时间戳
        Args:
            data_dirs: 数据目录字典
        Returns:
            list: 时间戳列表
        """
        # 从LiDAR目录获取时间戳
        lidar_files = os.listdir(data_dirs['lidar'])
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
        lidar_file = os.path.join(self.data_dirs['lidar'], f"{timestamp}.bin")
        if not os.path.exists(lidar_file):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_file}")
        
        # 读取二进制点云文件
        pointcloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, point_dim)
        return pointcloud

    def load_camera_image(self, camera_path, timestamp, camera_name):
        """
        加载相机图像
        Args:
            camera_path: 相机数据目录路径
            timestamp: 时间戳
            camera_name: 相机名称
        Returns:
            np.array: 图像数据
        """
        image_file = os.path.join(camera_path, camera_name, f"{timestamp}.jpg")
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file not found: {image_file}")
        
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

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

    def create_transformation_matrix(rotation, translation):
        """
        从旋转和平移创建变换矩阵
        Args:
            rotation: 旋转（四元数或旋转矩阵）
            translation: 平移向量
        Returns:
            np.array: 4x4变换矩阵
        """
        if isinstance(rotation, list) and len(rotation) == 4:
            # 四元数转换为旋转矩阵
            rot_matrix = Rotation.from_quat(rotation).as_matrix()
        elif isinstance(rotation, list) and len(rotation) == 9:
            # 3x3旋转矩阵
            rot_matrix = np.array(rotation).reshape(3, 3)
        else:
            raise ValueError("Unsupported rotation format")
        
        # 创建4x4变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = rot_matrix
        transformation[:3, 3] = np.array(translation)
        
        return transformation

    def transform_pointcloud(points, transformation_matrix):
        """
        使用变换矩阵变换点云
        Args:
            points: 点云数据 (N, 3)
            transformation_matrix: 4x4变换矩阵
        Returns:
            np.array: 变换后的点云
        """
        # 转换为齐次坐标
        points_hom = np.ones((points.shape[0], 4))
        points_hom[:, :3] = points[:, :3]
        
        # 应用变换
        transformed_points = points_hom.dot(transformation_matrix.T)
        
        return transformed_points[:, :3]

    # def project_lidar_to_camera(points, lidar_to_camera_matrix, camera_intrinsic):
    #     """
    #     将LiDAR点投影到相机坐标系
    #     Args:
    #         points: LiDAR点云 (N, 3)
    #         lidar_to_camera_matrix: LiDAR到相机的变换矩阵
    #         camera_intrinsic: 相机内参矩阵
    #     Returns:
    #         tuple: (图像坐标, 有效掩码)
    #     """
    #     # 转换到相机坐标系
    #     points_cam = transform_pointcloud(points, lidar_to_camera_matrix)
        
    #     # 过滤相机后面的点
    #     valid_mask = points_cam[:, 2] > 0.1
    #     points_cam_valid = points_cam[valid_mask]
        
    #     if len(points_cam_valid) == 0:
    #         return np.array([]), valid_mask
        
    #     # 投影到图像平面
    #     points_img_hom = points_cam_valid.dot(camera_intrinsic.T)
    #     points_img = points_img_hom[:, :2] / points_img_hom[:, 2:3]
        
    #     return points_img, valid_mask

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

    def generate_occupancy_grid(self,semantic_points, voxel_size, pc_range, occ_size):
        """
        从语义点云生成occupancy网格
        Args:
            semantic_points: 语义点云 (N, 4) [x, y, z, semantic]
            voxel_size: 体素大小
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            occ_size: occupancy网格大小
        Returns:
            tuple: (occupancy网格, 语义网格)
        """
        # 转换为体素坐标
        voxel_coords = np.floor((semantic_points[:, :3] - np.array(pc_range[:3])) / voxel_size).astype(int)
        
        # 创建网格
        occupancy_grid = np.zeros(occ_size, dtype=np.uint8)
        semantic_grid = np.zeros(occ_size, dtype=np.uint8)
        
        # 标记被占据的体素
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < occ_size[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < occ_size[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < occ_size[2])
        )
        
        valid_coords = voxel_coords[valid_mask]
        valid_semantics = semantic_points[valid_mask, 3]
        
        occupancy_grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = 1
        semantic_grid[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = valid_semantics
        
        return occupancy_grid, semantic_grid

    def process_occupancy_grid(self, occupancy_grid, semantic_grid, config):
        """
        后处理occupancy网格
        Args:
            occupancy_grid: 原始occupancy网格
            semantic_grid: 原始语义网格
            config: 配置参数
        Returns:
            tuple: 处理后的网格
        """
        processed_occupancy = occupancy_grid.copy()
        processed_semantic = semantic_grid.copy()
        
        # 应用形态学操作填充空洞
        if config.get('fill_holes', True):
            for z in range(occupancy_grid.shape[2]):
                slice_2d = occupancy_grid[:, :, z]
                filled_slice = ndimage.binary_fill_holes(slice_2d)
                processed_occupancy[:, :, z] = filled_slice.astype(np.uint8)
        
        # 应用高斯平滑
        if config.get('smooth_grid', True):
            sigma = config.get('smooth_sigma', 0.5)
            processed_occupancy = ndimage.gaussian_filter(
                processed_occupancy.astype(float), sigma=sigma
            )
            processed_occupancy = (processed_occupancy > 0.3).astype(np.uint8)
        
        return processed_occupancy, processed_semantic

    def create_dense_semantic_points(self, occupancy_grid, semantic_grid, voxel_size, pc_range):
        """
        从occupancy网格创建密集语义点云
        Args:
            occupancy_grid: occupancy网格
            semantic_grid: 语义网格
            voxel_size: 体素大小
            pc_range: 点云范围
        Returns:
            np.array: 密集语义点云
        """
        # 获取被占据的体素坐标
        occupied_indices = np.where(occupancy_grid > 0)
        
        if len(occupied_indices[0]) == 0:
            return np.array([])
        
        # 转换为世界坐标
        world_coords = np.stack(occupied_indices, axis=1).astype(float)
        world_coords[:, 0] = world_coords[:, 0] * voxel_size + pc_range[0]
        world_coords[:, 1] = world_coords[:, 1] * voxel_size + pc_range[1]
        world_coords[:, 2] = world_coords[:, 2] * voxel_size + pc_range[2]
        
        # 获取对应的语义标签
        semantics = semantic_grid[occupied_indices].reshape(-1, 1)
        
        # 合并坐标和语义
        dense_semantic_points = np.concatenate([world_coords, semantics], axis=1)
        
        return dense_semantic_points
    
    def create_voxel_occupancy(self, point_cloud):
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
        voxel = np.zeros(self.config['occ_size'])
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
        dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

        # to voxel coordinate
        pcd_np = dense_voxels_with_semantic
        pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / self.config['voxel_size']
        pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / self.config['voxel_size']
        pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / self.config['voxel_size']
        dense_voxels_with_semantic = np.floor(pcd_np).astype(np.int_)

        return dense_voxels_with_semantic
    def process_single_frame(self, timestamp):
        """
        处理单个时间戳的数据帧
        Args:
            timestamp: 时间戳
            data_dirs: 数据目录字典
            calibration_data: 标定数据
            label_mapping: 标签映射
            config: 配置参数
        Returns:
            dict: 处理结果
        """
        
        # 加载LiDAR点云
        lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])
        
        # 初始化语义点云
        semantic_points = np.zeros((lidar_points.shape[0], 4))
        semantic_points[:, :3] = lidar_points[:, :3]  # 坐标
        semantic_points[:, 3] = 0  # 默认语义标签
        
        # 处理每个相机
        camera_names = self.config['camera_types']
        
        for cam_name in camera_names:
            # 检查相机数据是否存在
            cam_dir = os.path.join(self.data_dirs['camera'], cam_name)
            if not os.path.exists(cam_dir):
                continue
            
            # 加载图像和mask
            image = self.load_camera_image(self.data_dirs['camera'], timestamp, cam_name)
            mask = self.load_semantic_mask(self.data_dirs['mask'], timestamp, cam_name)
            
            cam_calib = self.get_calibration(timestamp,cam_name)
            camera_matrix = cam_calib.get('camera_intrinsic', None)
            distortion_coeffs = cam_calib.get('distortion_coeffs', None)
            camera_to_ego_rotation = cam_calib.get('cam2ego_rotation', None)
            camera_to_ego_translation = cam_calib.get('cam2ego_translation', None)

            lidar_calib = self.get_calibration(timestamp,'lidar')
            lidar_to_ego_rotation = lidar_calib.get('lidar2ego_rotation', None)
            lidar_to_ego_translation = lidar_calib.get('lidar2ego_translation', None)

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

            np.save("debug.npy",points2d)

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
            save_path = os.path.join(self.data_dirs['save_path'],"{}.npy".format(timestamp))
            np.save(save_path, res)
        self.logger.info("------------ finish processing frame {} ------------".format(timestamp))

        return semantic_points
    
    
    # def generate_occupancy_for_sequence(self, timestamps, data_dirs, calibration_data, label_mapping, config, save_dir):
    #     """
    #     为整个序列生成occupancy真值
    #     Args:
    #         timestamps: 时间戳列表
    #         data_dirs: 数据目录字典
    #         calibration_data: 标定数据
    #         label_mapping: 标签映射
    #         config: 配置参数
    #         save_dir: 保存目录
    #     """
    #     # 创建保存目录
    #     os.makedirs(save_dir, exist_ok=True)
    #     os.makedirs(os.path.join(save_dir, 'occupancy_grids'), exist_ok=True)
    #     os.makedirs(os.path.join(save_dir, 'semantic_points'), exist_ok=True)
        
    #     # 处理每一帧
    #     all_world_points = []
        
    #     for timestamp in tqdm(timestamps, desc="Processing frames"):
    #         try:
    #             frame_result = process_frame(timestamp, data_dirs, calibration_data, label_mapping, config)
                
    #             # 保存原始语义点云
    #             np.save(
    #                 os.path.join(save_dir, 'semantic_points', f'{timestamp}_semantic.npy'),
    #                 frame_result['semantic_points']
    #             )
                
    #             # 累积世界坐标系的点云
    #             if len(frame_result['world_semantic_points']) > 0:
    #                 all_world_points.append(frame_result['world_semantic_points'])
                    
    #         except Exception as e:
    #             print(f"Error processing frame {timestamp}: {e}")
    #             continue
        
    #     if not all_world_points:
    #         print("No valid frames processed")
    #         return
        
    #     # 合并所有帧的点云
    #     combined_points = np.concatenate(all_world_points, axis=0)
        
    #     # 生成occupancy网格
    #     voxel_size = config['voxel_size']
    #     pc_range = config['pc_range']
    #     occ_size = config['occ_size']
        
    #     occupancy_grid, semantic_grid = generate_occupancy_grid(
    #         combined_points, voxel_size, pc_range, occ_size
    #     )
        
    #     # 后处理网格
    #     processed_occupancy, processed_semantic = process_occupancy_grid(
    #         occupancy_grid, semantic_grid, config
    #     )
        
    #     # 创建密集语义点云
    #     dense_points = create_dense_semantic_points(
    #         processed_occupancy, processed_semantic, voxel_size, pc_range
    #     )
        
    #     # 保存结果
    #     np.save(os.path.join(save_dir, 'occupancy_grid.npy'), processed_occupancy)
    #     np.save(os.path.join(save_dir, 'semantic_grid.npy'), processed_semantic)
        
    #     if len(dense_points) > 0:
    #         np.save(os.path.join(save_dir, 'dense_semantic_points.npy'), dense_points)
        
    #     print(f"Generated occupancy grid with {np.sum(processed_occupancy)} occupied voxels")


def main(config_path, data_root, save_path, label_map):
    """
    主函数
    Args:
        config_path: 配置文件路径
        data_root: 数据根目录
        save_path: 保存路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载标签映射
    with open(label_map, 'r') as f:
        label_mapping = yaml.safe_load(f)
    
    # 设置数据目录
    data_dirs = {
        'save_path': save_path,
        'calibration': os.path.join(data_root, 'calibration','calibration.json'),
        'camera':  os.path.join(data_root, 'camera'),
        'lidar': os.path.join(data_root, 'lidar'),
        'mask': os.path.join(data_root, 'masks'),
        'pose': os.path.join(data_root, 'pose', 'pose.json'),
        'label_mapping': label_map
    }
    autolabel = OCCAutolabel(data_dirs, config)
    
    # print(f"Processing {len(selected_timestamps)} frames from {start_idx} to {end_idx}")
    
    # 生成occupancy真值
    semantic_points = autolabel.process_single_frame('1532402927647951')
    print(semantic_points.shape)
    print(semantic_points[:10,:])
    print("--------------- finished ------------------")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate occupancy ground truth from custom dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--label_map', type=str, default=True, help='Label map file')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args.config, args.data_root, args.save_path, args.label_map)