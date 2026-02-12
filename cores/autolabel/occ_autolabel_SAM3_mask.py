import os
import glob
import re
import cv2
import yaml
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt

from .occ_autolabel_base import OCCAutolabelBase
from cores.Processor import PreProcessor, PostProcessor
from cores.category import Category

from tools.crop_partial_points import bin_to_pcd_xyz, bin_to_pcd

COLOR_MAP = {
    0: [21, 174, 103], # 绿色 静态物体
    1: [219, 79, 3], # 红色 车辆
    2: [250, 190, 0],   # 黄色  人
    3: [34, 174, 230], # 蓝色 路
    4: [255, 0, 0],
    5: [0, 0, 255]
}

def visualize_2d_points_mask(image, points2d, mask, alpha=0.5, point_color='red', point_size=2):
    """
    可视化图像、2D点和mask
    
    参数:
    - image: 原始图像 (H, W, 3) 或 (H, W)
    - points2d: 2D点坐标, 形状为 (N, 2) 或 (2, N)
    - mask: 分割mask, 形状为 (H, W)
    - alpha: mask的透明度
    - point_color: 点的颜色
    - point_size: 点的大小
    """
    
    # 确保图像是RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 确保mask是2D的
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    # 转换points2d为标准格式 (N, 2)
    points2d = np.array(points2d)
    if points2d.shape[0] == 2 and points2d.shape[1] > 2:
        points2d = points2d.T
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示带2D点和mask的图像
    axes[1].imshow(image)
    
    # 添加mask（半透明）
    # if mask is not None:
    #     mask_rgb = np.zeros_like(image)
    #     mask_rgb[:, :, 1] = mask * 255  # 绿色mask
    #     axes[1].imshow(mask_rgb, alpha=alpha)
    if mask is not None:
        # 创建彩色 mask 图像
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # 使用预定义的颜色映射
        for label, color in COLOR_MAP.items():
            if label != 0:
                mask_rgb[mask == label] = color
        
        axes[1].imshow(mask_rgb, alpha=alpha)
    
    # 添加2D点
    if points2d is not None and len(points2d) > 0:
        axes[1].scatter(points2d[:, 0], points2d[:, 1], 
                       c=point_color, s=point_size)
    
    axes[1].set_title('Image with 2D Points and Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def find_class_id_in_mask_file(search_dir):
    """
    在指定目录下查找mask开头，后缀为npy的文件，获取'_'后的数字作为class_id
    
    Args:
        search_dir: 搜索目录
    
    Returns:
        list: [(class_id, file_path)]
    """
    pattern = os.path.join(search_dir, "mask_*.npy")
    mask_files = glob.glob(pattern)
    
    file_path = mask_files[0]
    filename = os.path.basename(file_path)
    match = re.search(r'mask_(\d+)\.npy', filename)
    class_id = match.group(1)

    # if match:
    #     class_id = int(match.group(1))
    #     results.append((class_id, file_path))
    # for file_path in mask_files:
    #     filename = os.path.basename(file_path)
    #     match = re.search(r'mask_(\d+)\.npy', filename)
    #     if match:
    #         class_id = int(match.group(1))
    #         results.append((class_id, file_path))
    
    return class_id


class OCCAutolabelwithMask(OCCAutolabelBase):
    def __init__(self, config):
        super().__init__(config)
        self.pre_processor = PreProcessor(self.config)
        self.post_processor = PostProcessor(self.config)

    def inti_data(self, root):
        root_dirs = os.listdir(root)
        self.poses = []
        self.calibrations = {}
        self.timestamps = []
        self.label_mapping = {}
        self.lidar_path = ''
        self.camera_path = ''
        self.mask_path = ''
        self.timestamps = self.get_all_timestamps(os.path.join(root, 'lidar'))
        self.lidar_timestamps = self.get_all_timestamps(os.path.join(root, 'lidar'))
        self.lidar_path = os.path.join(root, 'lidar')

        self.camera_path = os.path.join(root, 'camera')

        self.camera_timestamps = self.get_camera_timestamps()

        # 读取标定文件：支持新的单独yaml格式
        calibration_dir = os.path.join(root, 'calibration')
        calibration_json = os.path.join(calibration_dir, 'calibration.json')
        
        if os.path.exists(calibration_json):
            # 旧格式：所有标定信息在一个json文件中
            self.calibrations = self.load_json_data(calibration_json)
        else:
            # 新格式：每个传感器单独的yaml文件
            self.calibrations = self._load_calibrations_from_yaml(calibration_dir)

        self.poses = self.load_json_data(os.path.join(root, 'pose', 'pose.json'))
        
        self.mask_path = os.path.join(root, 'mask_single_frame')
        
        if 'label_map' in self.config:
            with open(self.config['label_map'], 'r') as f:
                self.label_mapping = yaml.safe_load(f)

    
    def get_camera_timestamps(self):
        cam_names = os.listdir(self.camera_path)
        camera_timestamps = {}
        for cam_name in cam_names:
            cam_path = os.path.join(self.camera_path, cam_name)
            cam_timestamps = self.get_all_timestamps(cam_path)
            camera_timestamps[cam_name] = cam_timestamps
        return camera_timestamps
    
    def _load_calibrations_from_yaml(self, calibration_dir):
        """
        从lidar.yaml加载所有传感器标定信息
        
        Args:
            calibration_dir: 标定文件目录
        
        Returns:
            dict: 标定信息字典
        """
        calibrations = {}
        
        # 相机名称映射 (camera_front -> CAM_FRONT)
        camera_name_mapping = {
            'camera_front': 'CAM_FRONT',
            'camera_left': 'CAM_LEFT',
            'camera_right': 'CAM_RIGHT',
            'camera_back': 'CAM_BACK'
        }
        
        # 相机到lidar变换的映射
        camera_to_lidar_mapping = {
            'CAM_FRONT': 'camera_front_to_lidar_front',
            'CAM_LEFT': 'camera_left_to_lidar_front',
            'CAM_RIGHT': 'camera_right_to_lidar_front',
            'CAM_BACK': 'camera_back_to_lidar_front'
        }
        
        # 读取lidar.yaml
        lidar_yaml = os.path.join(calibration_dir, 'lidar.yaml')
        if not os.path.exists(lidar_yaml):
            raise FileNotFoundError(f"lidar.yaml不存在: {lidar_yaml}")
        
        # 读取YAML文件
        with open(lidar_yaml, 'r') as f:
            lidar_data = yaml.safe_load(f)
        
        # 获取内参和外参
        intrinsics = lidar_data.get('intrinsics', {})
        extrinsics = lidar_data.get('extrinsic_transforms', {})
        
        # 读取各个相机的标定信息
        for camera_key, cam_name in camera_name_mapping.items():
            camera_intrinsic = intrinsics.get(camera_key, {})
            
            if camera_intrinsic:
                # 读取K和D
                camera_matrix = np.array(camera_intrinsic['K'])
                distortion_coeffs = np.array(camera_intrinsic['D']).flatten()
                
                # 获取相机到lidar的变换
                camera_to_lidar_key = camera_to_lidar_mapping.get(cam_name, f"{camera_key}_to_lidar_front")
                camera_to_lidar = np.array(extrinsics.get(camera_to_lidar_key, np.eye(4)))
                
                # 存储标定信息
                calibrations[cam_name] = {
                    'camera_intrinsic': camera_matrix.tolist(),
                    'distortion_coeffs': distortion_coeffs.tolist(),
                    'camera_to_lidar': camera_to_lidar.tolist()
                }
        
        # 读取lidar变换链：lidar_front -> imu_front -> base
        lidar_front_to_imu_front = np.array(extrinsics.get('lidar_front_to_imu_front', np.eye(4)))
        imu_front_to_base = np.array(extrinsics.get('imu_front_to_base', np.eye(4)))
        lidar_to_ego_matrix = imu_front_to_base @ lidar_front_to_imu_front
        
        calibrations['lidar'] = {
            'lidar_to_ego': lidar_to_ego_matrix.tolist()
        }
        
        return calibrations
    
    def get_calibration(self, sensor_name):
        """
        获取传感器标定信息
        
        Args:
            sensor_name: 传感器名称
        
        Returns:
            dict: 标定信息
        """
        if 'lidar' in sensor_name.lower():
            lidar_to_ego = np.array(self.calibrations['lidar']['lidar_to_ego'])
            return {
                'lidar2ego_rotation': lidar_to_ego[:3, :3],
                'lidar2ego_translation': lidar_to_ego[:3, 3],
                'lidar_to_ego': lidar_to_ego
            }
        elif 'cam' in sensor_name.lower():
            calib = {
                'camera_intrinsic': np.array(self.calibrations[sensor_name]['camera_intrinsic']),
                'camera_to_lidar': np.array(self.calibrations[sensor_name]['camera_to_lidar'])
            }
            # 如果有畸变系数，也返回
            if 'distortion_coeffs' in self.calibrations[sensor_name]:
                calib['distortion_coeffs'] = np.array(self.calibrations[sensor_name]['distortion_coeffs'])
            return calib
        else:
            return {}

    def align_timestamps(self, cam_name, timestamp):
        """
        对齐相机时间戳和LiDAR时间戳，找到最接近的时间戳
        
        Args:
            cam_name: 相机名称
            timestamp: LiDAR时间戳
        
        Returns:
            str: 对齐后的相机时间戳
        """
        cam_timestamps = self.camera_timestamps.get(cam_name, [])
        if not cam_timestamps:
            return None
        
        # 将时间戳转换为整数进行比较
        target_time = int(timestamp)
        cam_times_int = [int(ts) for ts in cam_timestamps]
        
        # 找到最接近的时间戳
        closest_time = min(cam_times_int, key=lambda x: abs(x - target_time))

        if abs(closest_time - target_time) > 50000000:  # 超过0.05秒(50ms)则认为没有合适的时间戳
            return None
        
        return str(closest_time)

    def save_occupancy_to_ply(self, occupancy_data, ply_path):
        """
        将占用数据保存为PLY格式文件，将语义标签作为intensity存储
        
        Args:
            occupancy_data: 占用数据，形状为(N, 4)的数组[x, y, z, semantic_label]
            ply_path: PLY文件保存路径
        """
        import struct
        
        # 获取点的数量
        num_points = occupancy_data.shape[0]
        
        # 打开PLY文件进行写入
        with open(ply_path, 'wb') as f:
            # 写入PLY文件头
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float intensity
end_header
"""
            f.write(header.encode('utf-8'))
            
            # 写入点数据
            for i in range(num_points):
                x, y, z, semantic_label = occupancy_data[i]
                
                # 写入坐标和intensity（语义标签）
                f.write(struct.pack('<ffff', float(x), float(y), float(z), float(semantic_label)))
        
        self.logger.info(f"PLY文件已保存至: {ply_path}")



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
                semantics[i] = mask_label
        
        return semantics

    
    def assgin_semantic_from_image(self, timestamp, lidar_points=None):
        """
        通过点云向图片的投影, 将图像分割的mask语义赋予点云
        Args:
            timestamp: 时间戳
        Returns:
            dict: 处理结果
        """
        
        # 加载LiDAR点云
        if lidar_points is None:
            lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])
        
        # 初始化语义点云
        semantic_points = np.zeros((lidar_points.shape[0], 4))
        semantic_points[:, :3] = lidar_points[:, :3]  # 坐标
        semantic_points[:, 3] = Category.STATIC_OBJECT.value  # 默认语义标签
        
        # 处理每个相机
        camera_names = self.config['camera_types']
        
        for cam_name in camera_names:
            # 检查相机数据是否存在
            cam_dir = os.path.join(self.camera_path, cam_name)
            if not os.path.exists(cam_dir):
                continue
            
            # 加载图像和mask
            image_timestamp = self.align_timestamps(cam_name, timestamp)
            if image_timestamp is None:
                print(f"Warning: No aligned timestamp found for camera {cam_name} at LiDAR timestamp {timestamp}")
                continue
            # print(f"Using aligned timestamp {image_timestamp} for camera {cam_name} at LiDAR timestamp {timestamp}")
            # input("-----------")
            image = self.load_camera_image(image_timestamp, cam_name)
            mask = self.load_semantic_mask(self.mask_path, image_timestamp, cam_name)
            
            cam_calib = self.get_calibration(cam_name)
            camera_matrix = cam_calib.get('camera_intrinsic', None)
            distortion_coeffs = cam_calib.get('distortion_coeffs', None)
            camera_to_lidar = cam_calib.get('camera_to_lidar', None)
            
            # 将点云从ego坐标系转换到lidar_front坐标系
            # lidar_points当前是ego坐标系，需要转换到lidar_front坐标系进行投影
            # 因为projection中的camera_to_lidar是相机到lidar_front的变换
            lidar_calib = self.get_calibration('lidar')
            lidar_to_ego = lidar_calib['lidar_to_ego']
            ego_to_lidar = np.linalg.inv(lidar_to_ego)
            lidar_points_homo = self.projector.to_homo_coord(lidar_points[:, :3])
            lidar_points_in_lidar_front = (lidar_points_homo @ ego_to_lidar.T)[:, :3]

            # cv2.imread 获取的图片w和h顺序反向
            image_shape = np.zeros(2)
            image_shape[0], image_shape[1] = image.shape[1], image.shape[0]
            
            # 检查是否是鱼眼相机
            is_fisheye = False
            if distortion_coeffs is not None and len(distortion_coeffs) == 4:
                is_fisheye = True
            
            # 使用新的projection接口
            if is_fisheye:
                points2d, valid_mask = self.projector.projection_fisheye(
                    lidar_points_in_lidar_front,
                    image_shape=image_shape,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=distortion_coeffs,
                    camera_to_lidar=camera_to_lidar
                )
            else:
                points2d, valid_mask = self.projector.projection(
                    lidar_points_in_lidar_front,
                    image_shape=image_shape,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=distortion_coeffs,
                    camera_to_lidar=camera_to_lidar
                )

            # if cam_name == 'CAM_BACK':
            #     visualize_2d_points_mask(image, points2d, mask)

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
      
        return semantic_points

    def filter_dynamic_objects(self, semantic_points, offset_start, offset_end,
                               to_remove_labels=[1,2]):
        """
        过滤点云, 只保留offset_start-offset_end这段点云中的所有点,
        删除该范围外的动态对象点(汽车和行人)
        
        Args:
            semantic_points: np.ndarray - 点云数组，形状为(N, 4)的数组[x, y, z, semantic_label]
            offset_start: int - 起始索引
            offset_end: int - 结束索引
        
        Returns:
            np.ndarray - 过滤后的点云数组
        """
        # 创建一个全为True的掩码
        keep_mask = np.ones(semantic_points.shape[0], dtype=bool)
        
        # 对于范围外的点，只保留静态物体和路面点，过滤掉汽车和行人
        # 创建范围外点的掩码
        outside_range_mask = (np.arange(semantic_points.shape[0]) < offset_start) | \
                            (np.arange(semantic_points.shape[0]) >= offset_end)
        
        # 获取范围外点的语义标签
        outside_semantic_labels = semantic_points[outside_range_mask, 3]
        
        # 在范围外的点中，删除标签为汽车(1)和行人(2)的点
        dynamic_labels = to_remove_labels
        
        # 创建范围外动态物体的掩码
        dynamic_mask_outside = np.isin(outside_semantic_labels, dynamic_labels)
        
        # 将范围外动态物体点的keep_mask设置为False
        outside_indices = np.where(outside_range_mask)[0]
        keep_mask[outside_indices[dynamic_mask_outside]] = False
        
        # 应用掩码过滤点云
        filtered_points = semantic_points[keep_mask]
        
        return filtered_points
    
    def generate_dynamic_points(self, timestamp):
        """
        将当前帧中之前生成好的动态物体的点云放置到当前ego坐标系下
        """
        save_path = os.path.join(self.config['data_root'], 'dynamic_points', timestamp)
        all_dynamic_points = []
        camera_types = ["CAM_FRONT", "CAM_BACK", "CAM_LEFT", "CAM_RIGHT"]
        for cam_name in camera_types:
            dynamic_points_path = os.path.join(save_path, cam_name)
            if not os.path.exists(dynamic_points_path):
                continue
            instance_list = os.listdir(dynamic_points_path)
            for instance_id in instance_list:
                dynamic_points_bin_path = os.path.join(dynamic_points_path, instance_id, "dense_points_processed.bin")
                if not os.path.exists(dynamic_points_bin_path):
                    continue
                    dynamic_points_bin_path = os.path.join(dynamic_points_path, instance_id, "dense_points_processed_fulfilled.bin")
                    if not os.path.exists(dynamic_points_bin_path):
                        continue
                dynamic_points = np.fromfile(dynamic_points_bin_path, dtype=np.float32).reshape(-1, 3)
                # 从mask文件名中获取类别ID
                class_id = find_class_id_in_mask_file(os.path.join(dynamic_points_path, instance_id))
                class_id = int(class_id)
                semantic_info = np.ones(dynamic_points.shape[0], dtype=np.int32) * class_id

                dynamic_points_with_semantics = np.concatenate([dynamic_points, semantic_info[:, np.newaxis]], axis=1)

                all_dynamic_points.append(dynamic_points_with_semantics)
        
        if len(all_dynamic_points) == 0:
            return None
        
        all_dynamic_points = np.concatenate(all_dynamic_points, axis=0)

        return all_dynamic_points

        

    def process_multi_frame(self):
        semantic_points = []
        for i, timestamp in enumerate(tqdm(self.timestamps)):
            lidar_points = self.load_lidar_pointcloud(timestamp,self.config['point_dim'])[:,:3]

            # 将点云从lidar坐标系转换到ego坐标系
            calib = self.get_calibration('lidar')
            lidar_to_ego = calib['lidar_to_ego']
            lidar_points_homo = self.projector.to_homo_coord(lidar_points)
            points_on_ego = lidar_points_homo @ lidar_to_ego.T

            # -------------------------------------- 去除自车点云 --------------------------------------
            points_without_ego = self.pre_processor.remove_ego_points(points_on_ego)
            
            # -------------------------------------- 赋予点云语义 --------------------------------------
            sem_point = self.assgin_semantic_from_image(timestamp, points_without_ego)

            # -------------------------------------- 地面盲区赋值地面语义 --------------------------------------
            sem_point = self.pre_processor.redefine_blind_floor_points(sem_point)

            # -------------------------------------- 从ego坐标系转到世界坐标系下 --------------------------------------
            coords = sem_point[:, :3]
            current_pose = self.get_pose(timestamp)
            current_position = np.array(current_pose['translation'])
            current_quaternion = np.array(current_pose['rotation'])
            # pose文件中是lidar_front的pose，需要转换为ego的pose
            lidar_to_world = self.projector.to_matrix4x4(current_quaternion, current_position)
            # ego_to_world = lidar_to_world @ inv(lidar_to_ego)
            ego_to_world = lidar_to_world @ np.linalg.inv(lidar_to_ego)
            
            points_homo = self.projector.to_homo_coord(coords)
            p_in_world = (points_homo @ ego_to_world.T)[:, :3]
            p_in_world = np.hstack((p_in_world, sem_point[:,3:]))

            semantic_points.append(p_in_world)
        
        # static_points = self.post_processor.full_point_process(semantic_points)
        static_points = np.vstack(semantic_points)

        # save_path = "/home/robot/data/Autolabel/AUTOLABEL_0205/debug/clip0006/full_points_world.pcd"
        # to_saved_points = np.vstack(static_points)
        # bin_to_pcd(to_saved_points, save_path)
        # input("------------------------ yeah")

        # 获取dynamic_points目录下的所有timestamp文件夹
        dynamic_points_dir = os.path.join(self.config['data_root'], 'dynamic_points')
        if os.path.exists(dynamic_points_dir):
            to_generated_timestamps = sorted([d for d in os.listdir(dynamic_points_dir) 
                                             if os.path.isdir(os.path.join(dynamic_points_dir, d))])
        else:
            to_generated_timestamps = self.timestamps
            print(f"Warning: dynamic_points directory not found at {dynamic_points_dir}, using self.timestamps instead")
        
        # 将全量点云转到每一帧的ego坐标系下并计算相应的Occupacy Voxel
        for i, tp in enumerate(to_generated_timestamps):
            ref_pose = self.get_pose(tp)
            ref_position = np.array(ref_pose['translation'])
            ref_quaternion = np.array(ref_pose['rotation'])
            # pose文件中是lidar_front的pose，需要转换为ego的pose
            lidar_to_world = self.projector.to_matrix4x4(ref_quaternion, ref_position)
            ego_to_world = lidar_to_world @ np.linalg.inv(lidar_to_ego)

            static_semantic_info = static_points[:, 3]
            static_points_xyz = static_points[:, :3]
            static_points_xyz_homo = self.projector.to_homo_coord(static_points_xyz)
            transform_matrix = np.linalg.inv(ego_to_world)
            static_points_in_ego = static_points_xyz_homo @ transform_matrix.T[:, :3]
            static_semantic_points_full = np.hstack((static_points_in_ego, static_semantic_info.reshape(-1, 1)))

            # -------------------------------------- 对ego下的多帧点云进行后处理 --------------------------------------
            static_semantic_points_full = self.post_processor.points_in_range(static_semantic_points_full)

            static_semantic_points_full = self.post_processor.post_process_on_ego(static_semantic_points_full)

            # -------------------------------------- 加入动态点云 --------------------------------------
            dynamic_semantic_points_full = self.generate_dynamic_points(tp)

            final_points = self.post_processor.points_in_range(static_semantic_points_full)
            if dynamic_semantic_points_full is not None:
                dynamic_semantic_points_full = self.post_processor.points_in_range(dynamic_semantic_points_full)
                final_points = np.vstack([dynamic_semantic_points_full,static_semantic_points_full])

            # -------------------------------------- 动态点云去噪 --------------------------------------
            final_points = self.post_processor.dynamic_points_denoise(final_points)
            # res = self.create_voxel_occupancy_slim(final_static_poinst)
            res =self.create_voxel_occupancy_dense(final_points)

            # save_path2 = "/home/robot/data/Autolabel/AUTOLABEL_0205/debug/clip0006/occ_result.pcd"
            # bin_to_pcd(res, save_path2)
            # input("------------------------ yeah2")

            res = self.post_processor.calculate_free_unknown(res)

            # save to npz
            os.makedirs(self.config['save_path'], exist_ok=True)
            save_path = os.path.join(self.config['save_path'],"{}.npz".format(tp))
            res = {'occ_gt': res}
            np.savez(save_path, **res)

            # save to ply
            # res_occupied = res[res[:,3]!=100]
            # save_path_ply = "/home/robot/data/nuscenes_mini_clips/clip0000/occ_gt_ply"
            # save_path_ply = os.path.join(save_path_ply, "{}.ply".format(tp))
            # self.save_occupancy_to_ply(res_occupied, save_path_ply)

            print("--------------------- finish processing frame {}".format(tp))