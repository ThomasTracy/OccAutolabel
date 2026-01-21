import os
import cv2
import yaml
import numpy as np
from typing import List, Dict, Tuple

from .occ_autolabel_base import OCCAutolabelBase


COLOR_MAP = {
    0: [0, 0, 0],
    1: [47, 254, 138],
    2: [120, 120, 120],
    3: [246, 247, 84],
    4: [255, 0, 0],
    5: [0, 0, 255]
}
def draw_points_on_image(image: np.ndarray, points_2d: np.ndarray, 
                        point_size: int = 5, 
                        semantic: np.ndarray = None,
                        color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        在图像上绘制投影点
        
        Args:
            image: 原始图像
            points_2d: 投影点坐标
            point_size: 点大小
            color: 点颜色 (B, G, R)
            
        Returns:
            带投影点的图像
        """
        image_with_points = image.copy()
        overlay = image_with_points.copy()
        
        for i, point in enumerate(points_2d):
            if semantic is not None:
                color = COLOR_MAP[semantic[i]]
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(overlay, (x, y), point_size, color, -1)
        
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, image_with_points, 1-alpha, 0, image_with_points)
        
        return image_with_points

class OCCAutolabelwithMask(OCCAutolabelBase):
    def __init__(self, config):
        super().__init__(config)

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
        self.timestamps = self.get_all_timestamps(os.path.join(root, 'mask', 'CAM_FRONT'))
        self.lidar_timestamps = self.get_all_timestamps(os.path.join(root, 'lidar'))
        self.lidar_path = os.path.join(root, 'lidar')

        self.camera_path = os.path.join(root, 'camera')

        self.calibrations = self.load_json_data(os.path.join(root, 'calibration', 'calibration.json'))

        self.poses = self.load_json_data(os.path.join(root, 'pose', 'pose.json'))
        
        self.mask_path = os.path.join(root, 'mask')
        
        if 'label_map' in self.config:
            with open(self.config['label_map'], 'r') as f:
                self.label_mapping = yaml.safe_load(f)

    def split_lidar_batches(self):
        lidar_batches = []
        cur_batch = []
        previous_tp = None
        for tp in self.lidar_timestamps:
            if previous_tp is None:
                cur_batch.append(tp)
            else:
                if abs(float(previous_tp) - float(tp)) > 10000000:
                    lidar_batches.append(cur_batch)
                    cur_batch = []
                cur_batch.append(tp)
            previous_tp = tp
        return lidar_batches

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

            # res_imag = draw_points_on_image(image, points2d, valid_mask)

            # save image
            # save_path = "res_img.png"
            # cv2.imwrite(save_path, res_imag)
            # input(".................")

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
            # save_path = "semantic_points.bin"
            # with open(save_path, 'wb') as f:
            #     semantic_points.astype(np.float32).tofile(f)
            # input("semantic points.................")
            res = self.create_voxel_occupancy_slim(semantic_points)
            os.makedirs(self.config['save_path'], exist_ok=True)
            save_path = os.path.join(self.config['save_path'],"{}.npz".format(timestamp))
            res = {'occ_gt': res}
            np.savez(save_path, **res)
      
        return semantic_points

    def process_multi_frame(self, debug_timestamp=None):
        for i, timestamp in enumerate(self.timestamps):
            if debug_timestamp is not None and timestamp != debug_timestamp:
                continue
            frame_radius = self.config['multi_frame_radius']
            selected_poses = []
            selected_points = []
            ref_pose = self.get_pose(timestamp)

            tp_batches = self.split_lidar_batches()
            selected_batch = None
            for batch in tp_batches:
                if float(timestamp) >= float(batch[0]) and float(timestamp) < float(batch[-1]):
                    selected_batch = batch
                    break

            if 2 * frame_radius + 1 >= len(selected_batch):
                selected_tp = selected_batch
            start_idx = max(0, i - frame_radius)
            end_idx = min(len(selected_batch), i + frame_radius + 1)
            if start_idx == 0:
                end_idx = 2 * frame_radius + 1
            if end_idx == len(selected_batch):
                start_idx = len(selected_batch) - 2 * frame_radius - 1
            selected_tp = selected_batch[start_idx:end_idx]
            for tp in selected_tp:
                selected_poses.append(self.get_pose(tp))
                lidar_points = self.load_lidar_pointcloud(tp,self.config['point_dim'])[:,:3]

                # 将点云从lidar坐标系转换到ego坐标系
                calib = self.get_calibration(tp,'lidar')
                lidar_to_ego = self.projector.to_matrix4x4(calib['lidar2ego_rotation'],
                                                        calib['lidar2ego_translation'])
                lidar_points_homo = self.projector.to_homo_coord(lidar_points)
                points_on_ego = lidar_points_homo @ lidar_to_ego.T
                # 去除自车点云
                mask = (points_on_ego[:, 0] > 3) | (points_on_ego[:, 0] < -1) | \
                        (points_on_ego[:, 1] > 1) | (points_on_ego[:, 1] < -1)
                points_on_ego = points_on_ego[mask]

                # save_path = "points_on_ego.bin"
                # if not os.path.exists(save_path):
                #     with open(save_path, 'wb') as f:
                #         points_on_ego.astype(np.float32).tofile(f)
                # input("points_on_ego.................")

                selected_points.append(points_on_ego[:,:3])
            
            selected_points = self.projector.multi_lidar_concat(selected_points, selected_poses, ref_pose)
            # semantic_points = np.zeros((lidar_points.shape[0], 4))
            # semantic_points[:, :3] = lidar_points[:, :3]  # 坐标
            # semantic_points[:, 3] = 0  # 默认语义标签
            # save_path = "full_points.bin"
            # if not os.path.exists(save_path):
            #     with open(save_path, 'wb') as f:
            #         semantic_points.astype(np.float32).tofile(f)
            # input("Press Enter to continue...")
            self.process_single_frame(timestamp, selected_points)
            if debug_timestamp is not None:
                return
            self.logger.info("------------ finish processing frame {} ------------".format(timestamp))