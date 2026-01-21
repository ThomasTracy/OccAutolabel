import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import json
from typing import List, Dict, Tuple
import os
import glob
import pandas as pd
import traceback
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from pyquaternion import Quaternion

CAM_TRANS = np.array([
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]
                    ])

COLOR_MAP = {
    0: [0, 0, 0],
    1: [47, 254, 138],
    2: [120, 120, 120],
    3: [246, 247, 84],
    4: [255, 0, 0],
    5: [0, 0, 255]
}

INTENSITY_TO_LABEL = {
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    50: 5,
    255: 0
}

class PointCloudProjector:
    def __init__(self, calib_path: str):
        """
        初始化点云投影器
        
        Args:
            calib_path: 标定文件路径
        """
        self.calib_data = self.load_calibration(calib_path)
    
    def to_matrix4x4(self, rotation, translation=None):
        '''
        将旋转平移矩阵转换成4x4矩阵
        '''
        # 注意此处输入的四元数是什么顺序，是[w, x, y, z]还是[x, y, z, w]
        # pyquatanion中，是[w, x, y, z]
        # scipy.spatial.transform.Rotation.from_quat 中是[x, y, z, w]
        if rotation.shape == (3,):
            rotation = R.from_euler('xyz', rotation, degrees=True).as_matrix()
        elif rotation.shape == (4,):
            rotation = Quaternion(rotation).rotation_matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        return transformation_matrix
    def to_homo_coord(self, points):
        '''
        将点云转换成齐次坐标
        '''
        return np.hstack((points, np.ones((points.shape[0], 1))))
    def load_calibration(self, calib_path: str) -> Dict:
        """
        加载相机和LiDAR的标定数据
        
        Args:
            calib_path: 标定文件路径
            
        Returns:
            标定数据字典
        """
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        return calib_data
    
    def lidar_to_ego(self, points: np.ndarray) -> np.ndarray:
        ego2lidar_rotation = np.array(self.calib_data['lidar']['rotation'])
        ego2lidar_translation = np.array(self.calib_data['lidar']['translation'])
        ego2lidar_transform = self.to_matrix4x4(ego2lidar_rotation, ego2lidar_translation)
        lidar2ego_transform = np.linalg.inv(ego2lidar_transform)
        return points @ lidar2ego_transform.T

    def lidar_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """
        将点云从LiDAR坐标系转换到相机坐标系
        
        Args:
            points: 点云数据 (N, 3)
            camera_name: 相机名称 ('front', 'back', 'left', 'right')
            
        Returns:
            转换后的点云 (N, 3)
        """
        if camera_name == 'left':
            a = 0
        camera_name = "camera_" + camera_name
        # 获取外参矩阵 (LiDAR到相机的变换)
        ego2camera_rotation = np.array(self.calib_data[camera_name]['rotation'])
        ego2camera_translation = np.array(self.calib_data[camera_name]['translation'])
        ego2camera_transform = self.to_matrix4x4(ego2camera_rotation, ego2camera_translation)

        # 相机坐标系有一个旋转
        ego2camera_transform = CAM_TRANS @ ego2camera_transform
        
        # 应用外参变换
        points_camera = points @ ego2camera_transform.T
        
        return points_camera[:, :3]
    

    def project_to_image_fisheye(self, camera_points, camera_name):

        camera_name = "camera_" + camera_name

        camera_matrix = np.array(self.calib_data[camera_name]['intrinsic']).reshape(3, 3)
        dist_coeffs = np.array(self.calib_data[camera_name]['distortion_coeffs'])
        
        X = camera_points[:, 0]
        Y = camera_points[:, 1]
        Z = camera_points[:, 2]
        
        valid_mask = Z > 0

        Xc = X[valid_mask]
        Yc = Y[valid_mask]
        Zc = Z[valid_mask]
        
        # 步骤2: 投影到归一化平面
        x = Xc / Zc
        y = Yc / Zc
        
        # 步骤3: 应用鱼眼畸变模型 (Kannala-Brandt模型)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r)
        
        # 提取畸变系数
        k1, k2, k3, k4 = dist_coeffs.flatten()[:4]
        
        # 计算畸变半径
        theta2 = theta * theta
        theta3 = theta2 * theta
        theta5 = theta3 * theta2
        theta7 = theta5 * theta2
        theta9 = theta7 * theta2
        
        r_d = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9
        
        # 计算畸变后的归一化坐标
        scaling = r_d / (r + 1e-6)
        x_d = x * scaling
        y_d = y * scaling
        
        # 步骤4: 归一化平面 -> 像素坐标系
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        u = fx * x_d + cx
        v = fy * y_d + cy

        res = np.stack([u, v], axis=1)
        
        return res, valid_mask

    def project_to_image(self, points_camera: np.ndarray, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        将相机坐标系下的点云投影到图像平面
        
        Args:
            points_camera: 相机坐标系下的点云 (N, 3)
            camera_name: 相机名称
            
        Returns:
            (投影后的2D坐标, 有效的点云索引)
        """
        camera_name = "camera_" + camera_name
        # 获取内参矩阵
        intrinsic = np.array(self.calib_data[camera_name]['intrinsic']).reshape(3, 3)
        # distortion = np.array(self.calib_data[camera_name]['distortion_coeffs'])
        distortion = np.zeros(5)
        
        # 过滤掉相机后面的点 (z <= 0)
        valid_mask = points_camera[:, 2] > 0
        points_valid = points_camera[valid_mask]
        
        if len(points_valid) == 0:
            return np.array([]), valid_mask
        
        point_on_image = points_valid @ intrinsic.T
        point_on_image[:,0] = point_on_image[:,0] / (point_on_image[:,2] + 1e-8)
        point_on_image[:,1] = point_on_image[:,1] / (point_on_image[:,2] + 1e-8)

        # 提取2D像素坐标（取整到整数像素位置）
        pixel_coords = np.round(point_on_image[:, :2]).astype(int)
        
        # 为每个点创建包含像素坐标和深度信息的结构
        points_with_depth = np.column_stack((pixel_coords, points_valid[:, 2]))  # (x, y, z)
        
        # 按像素坐标分组，并保留每个像素中Z值最小的点（最近的）
        pixel_dict = {}
        for i, (x, y, z) in enumerate(points_with_depth):
            pixel_key = (x, y)
            if pixel_key not in pixel_dict or z < pixel_dict[pixel_key][1]:
                pixel_dict[pixel_key] = (i, z)  # 存储原始索引和深度
        
        # 获取所有需要保留的点的索引
        keep_indices = [idx for idx, _ in pixel_dict.values()]
        
        # 创建新的有效掩码
        new_valid_mask = np.zeros_like(valid_mask)
        original_valid_indices = np.where(valid_mask)[0]
        new_valid_mask[original_valid_indices[keep_indices]] = True
        
        # 返回过滤后的投影点和新的有效掩码
        filtered_points = point_on_image[keep_indices]
            
        # 应用内参投影
        # point_on_image, _ = cv2.projectPoints(
        #                                         points_valid,
        #                                         np.eye(3),
        #                                         np.zeros((3, 1)),
        #                                         intrinsic,
        #                                         distortion
        #                                     )
        
        return point_on_image, valid_mask
    
    def filter_points_in_image(self, points_2d: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        过滤在图像边界内的点
        
        Args:
            points_2d: 投影点 (N, 2)
            image_shape: 图像形状 (height, width)
            
        Returns:
            在图像内的点索引
        """
        if len(points_2d) == 0:
            return np.array([])
        
        height, width = image_shape
        in_image_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                       (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        
        return in_image_mask

class ImageStitcher:
    def __init__(self, output_size: Tuple[int, int] = (1920, 1080)):
        """
        初始化图像拼接器
        
        Args:
            output_size: 输出图像尺寸 (width, height)
        """
        self.output_size = output_size
        
    def stitch_images(self, images: Dict[str, np.ndarray], voxel_image) -> np.ndarray:
        """
        将四张图像拼接成一张
        
        Args:
            images: 图像字典 {'front': img, 'back': img, 'left': img, 'right': img}
            
        Returns:
            拼接后的图像
        """
        # 调整图像大小为输出尺寸的1/3
        target_size = (self.output_size[0] // 3, self.output_size[1] // 3)
        resized_images = {}
        
        for camera_name, img in images.items():
            resized_images[camera_name] = cv2.resize(img, target_size)

        resized_voxel_image = cv2.resize(voxel_image, target_size)  
        
        # 创建拼接画布 (3行3列的布局)
        stitched = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        # 布局: 三行三列
        # [    ][前向][    ]
        # [左向][voxel][右向]
        # [    ][后向][    ]
        h, w = target_size[1], target_size[0]
        
        # 前相机 - 第一行中央
        if 'front' in resized_images:
            stitched[0:h, w:w*2] = resized_images['front']
        # 后相机 - 第二行中央
        if 'back' in resized_images:
            stitched[h*2:h*3, w:w*2] = resized_images['back']
        # 左相机 - 第二行左侧
        if 'left' in resized_images:
            stitched[h:h*2, 0:w] = resized_images['left']
        # 右相机 - 第二行右侧
        if 'right' in resized_images:
            stitched[h:h*2, w*2:w*3] = resized_images['right']

        stitched[h:h*2, w:w*2] = resized_voxel_image
        
        return stitched

class PointCloudVisualizer:
    def __init__(self, calib_path: str, 
                 odom_path: str,
                 output_size: Tuple[int, int] = (1920, 1080)):
        """
        初始化点云可视化器
        
        Args:
            calib_path: 标定文件路径
            output_size: 输出图像尺寸
        """
        self.projector = PointCloudProjector(calib_path)
        self.stitcher = ImageStitcher(output_size)
        self.poses = self.load_csv_data(odom_path)
    
    def load_csv_data(self, odom_path):
        """
        加载里程计数据
        CSV格式为: timestamp translation, angular
        """
        odom_df = pd.read_csv(odom_path)
        
        # 检查必要的列是否存在
        required_columns = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for col in required_columns:
            if col not in odom_df.columns:
                raise ValueError(f"里程计文件缺少必要的列: {col}")
            
        result = {}
        for _, row in odom_df.iterrows():
            timestamp = str(int(row['timestamp']))
            result[timestamp] = {
                'translation': [
                    float(row['x']),
                    float(row['y']),
                    float(row['z'])
                ],
                'rotation': [
                    float(row['roll']),
                    float(row['pitch']),
                    float(row['yaw'])
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
    

    def point_downsample(self, point_cloud, voxel_size=0.1, num_classes=6):
        '''
        直接对点云进行体素化，得到每个体素的中心点
        体素的语义为每个体素中最多的点的语义
        '''
        pc_xyz = point_cloud[:, :3]
        pc_semantic = point_cloud[:, 3]
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
    
    def process_single_frame(self, point_cloud_path: str, 
                             image_paths: Dict[str, str],
                             voxel_image_path: str,
                             timestamp) -> np.ndarray:
        """
        处理单帧数据
        
        Args:
            point_cloud_path: 点云文件路径
            image_paths: 图像路径字典
            
        Returns:
            拼接后的带投影点的图像
        """
        # 加载点云
        point_cloud = self.load_point_cloud(point_cloud_path)
        point_cloud = self.point_downsample(point_cloud)
        # point_cloud = np.array([
        #     [17.4758, -2.7357, 1.1572, 1]
        # ])
        
        current_pose = self.get_pose(timestamp)
        current_position = np.array(current_pose['translation'])
        current_quaternion = np.array(current_pose['rotation'])
        lidar_to_world = self.projector.to_matrix4x4(current_quaternion, current_position)
        world_to_lidar = np.linalg.inv(lidar_to_world)

        point_cloud_xyz = point_cloud[:, :3]
        semantic = point_cloud[:, 3]
        intensity2label = INTENSITY_TO_LABEL
        for k,v in intensity2label.items():
            semantic[semantic == k] = v

        points_homo = self.projector.to_homo_coord(point_cloud_xyz)
        point_on_lidar = points_homo @ world_to_lidar.T
        point_on_ego = self.projector.lidar_to_ego(point_on_lidar)
        mask = (point_on_ego[:, 0] > -20) &  (point_on_ego[:, 0] < 20) & \
                (point_on_ego[:, 1] > -20) & (point_on_ego[:, 1] < 20) & \
                (point_on_ego[:, 2] > -1) & (point_on_ego[:, 2] < 5)
        saved_points = point_on_ego[mask]
        semantic = semantic[mask]

        full_points_in_world = np.concatenate([saved_points[:,:3], semantic[:, None]], axis=1)
        save_path = os.path.join('/home/jszr/Data/OCC_Autolabel/DATA/ts_align_debug', 'points_ego.bin')
        # if not os.path.exists(save_path):
        with open(save_path, 'wb') as f:
            full_points_in_world.astype(np.float32).tofile(f)

        images_with_points = {}
        
        # 对每个相机进行处理
        for camera_name, image_path in image_paths.items():
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Cannot load image {image_path}")
                continue
                
            # 投影点云到图像
            points_camera = self.projector.lidar_to_camera(saved_points, camera_name)
            # points_2d, mask_front = self.projector.project_to_image_fisheye(points_camera, camera_name)
            points_2d, mask_front = self.projector.project_to_image(points_camera, camera_name)
            # points_2d = points_2d.reshape(-1, 2)
            valid_semantic = semantic[mask_front]
            
            if len(points_2d) > 0:
                # 过滤在图像内的点
                in_image_mask = self.projector.filter_points_in_image(points_2d, image.shape[:2])
                valid_points_2d = points_2d[in_image_mask]
                valid_semantic = valid_semantic[in_image_mask]

                # 在图像上绘制投影点
                image_with_points = self.draw_points_on_image(image, valid_points_2d, semantic=valid_semantic)
            else:
                image_with_points = image.copy()
                
            images_with_points[camera_name] = image_with_points
        
        if voxel_image_path is not None:
            voxel_image = cv2.imread(voxel_image_path)
        else:
            voxel_image = np.zeros((500, 500, 3), dtype=np.uint8)

        # 拼接图像
        stitched_image = self.stitcher.stitch_images(images_with_points,voxel_image)
        
        return stitched_image
    
    def load_point_cloud(self, file_path: str) -> np.ndarray:
        """
        加载点云数据
        
        Args:
            file_path: 点云文件路径
            
        Returns:
            点云数据 (N, 3)
        """
        # 支持多种点云格式
        if file_path.endswith('.bin'):
            # KITTI格式的bin文件
            point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            return point_cloud
        elif file_path.endswith('.pcd'):
            # PCD文件
            pcd = o3d.io.read_point_cloud(file_path)
            return np.asarray(pcd.points)
        elif file_path.endswith('.npy'):
            # numpy数组
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {file_path}")
    
    def draw_points_on_image(self, image: np.ndarray, points_2d: np.ndarray, 
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
    
    def draw_timestamp_on_image(self, image, tp_diff, cams_tp):
        """
        在图像上绘制时间戳
        
        Args:
            image: 原始图像
            tp_diff: 时间戳差
            cams_tp: 相机时间戳
        """
        # 准备要显示的文本内容
        text_lines = []
        for cam_name in cams_tp.keys():
            line = f"{cam_name}: {cams_tp[cam_name]} (diff: {tp_diff[cam_name]:.2f}ms)"
            text_lines.append(line)
        
        # 设置字体和尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)  # 白色
        thickness = 2
        bg_color = (0, 0, 0)  # 黑色背景
        
        # 计算文本区域位置（图像中间偏上）
        height, width = image.shape[:2]
        # start_x = 640 + 50
        # start_y = 360 + 50
        start_x = 0 + 50
        start_y = 0 + 50
        
        # 逐行绘制文本
        for i, line in enumerate(text_lines):
            y_pos = start_y + i * 30
            # 添加文本背景以提高可读性
            # (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            # cv2.rectangle(image, 
            #             (start_x - 5, y_pos - text_height - 5), 
            #             (start_x + text_width + 5, y_pos + baseline + 5), 
            #             bg_color, 
            #             -1)
            cv2.putText(image, line, (start_x, y_pos), font, font_scale, color, thickness)

        return image

def generate_video_from_frames(frame_folder: str, output_video_path: str, fps: int = 10):
    """
    从帧图像生成视频
    
    Args:
        frame_folder: 帧图像文件夹
        output_video_path: 输出视频路径
        fps: 帧率
    """
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(('.png', '.png'))])
    
    if not frame_files:
        print("No frame images found!")
        return
    
    # 获取第一帧的尺寸
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Generating video with {len(frame_files)} frames...")
    
    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to: {output_video_path}")


def find_closest_timestamp(timestamps, target_timestamp):
    if not timestamps:
        return None
    closest_timestamp = min(timestamps, key=lambda x: abs(float(x) - float(target_timestamp)))
    return closest_timestamp

def extract_timestamp(file_path):
    timestamps = [os.path.splitext(os.path.basename(file_path))[0] for file_path in os.listdir(file_path)]
    if len(timestamps) == 0:
        return [] 
    return sorted(timestamps)

def align_timestamps(cam_ts_dict, lidar):
    lidar_ts = float(lidar) / 1e6
    res = {}
    tp_diff = {}
    cam_tp = {}
    for k in cam_ts_dict:
        cam_ts = float(cam_ts_dict[k]) / 1e6
        if abs(cam_ts - lidar_ts) < 0.01:
            res[k] = cam_ts_dict[k]
            tp_diff[k] = (cam_ts - lidar_ts) * 1000
            cam_tp[k] =cam_ts_dict[k]
    
    return res, tp_diff, cam_tp

def main(root):
    """
    主函数 - 示例使用方法
    """
    # 配置路径
    world_points_path = os.path.join(root, "world_points.bin")
    calib_path = os.path.join(root, "calibration.json")  # 标定文件路径
    odom_path = os.path.join(root, "odometry.csv")
    point_cloud_dir = os.path.join(root, "lidar")  # 点云文件夹
    image_dir = os.path.join(root, "images")  # 图像文件夹
    gt_npz_dir = os.path.join(root, "annotations", "vis_frames")
    output_dir = os.path.join(root, "video")  # 输出文件夹
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化可视化器
    visualizer = PointCloudVisualizer(calib_path, odom_path)
    
    cam_front_timestamps = extract_timestamp(os.path.join(image_dir, "front"))
    cam_back_timestamps = extract_timestamp(os.path.join(image_dir, "back"))
    cam_left_timestamps = extract_timestamp(os.path.join(image_dir, "left"))
    cam_right_timestamps = extract_timestamp(os.path.join(image_dir, "right"))
    lidar_timestamps = extract_timestamp(point_cloud_dir)

    
    to_process_timestamps = lidar_timestamps[:]
    # to_process_timestamps = ['1762160768145748']
    print("Processing frames...")
    for ts in tqdm(to_process_timestamps):
        # if ts != '1762941737009106':
        #     continue
        # 构建文件路径
        if not os.path.exists(world_points_path):
            point_cloud_path = f"{point_cloud_dir}/{ts}.bin"
        else:
            point_cloud_path = world_points_path
        cam_front_ts = find_closest_timestamp(cam_front_timestamps, ts)
        cam_back_ts = find_closest_timestamp(cam_back_timestamps, ts)
        cam_left_ts = find_closest_timestamp(cam_left_timestamps, ts)
        cam_right_ts = find_closest_timestamp(cam_right_timestamps, ts)

        cam_ts_dict = {}
        if cam_front_ts is not None:
            cam_ts_dict["front"] = cam_front_ts
        if cam_back_ts is not None:
            cam_ts_dict["back"] = cam_back_ts
        if cam_left_ts is not None:
            cam_ts_dict["left"] = cam_left_ts
        if cam_right_ts is not None:
            cam_ts_dict["right"] = cam_right_ts

        algined_ts, tp_diff, cams_tp = align_timestamps(cam_ts_dict, ts)
        image_paths = {}
        for k, v in algined_ts.items():
            image_paths[k] = f"{image_dir}/{k}/{v}.png"

        if len(image_paths) == 0:
            print(f"Warning: No images found for frame {ts}")
            continue

        # 体素可视化结果的Path
        voxel_image_path = f"{gt_npz_dir}/{ts}.png"
        if not os.path.exists(voxel_image_path):
            voxel_image_path = None
        
        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [point_cloud_path] + list(image_paths.values())):
            print(f"Warning: Missing files for frame {ts}")
            continue
        
        # 处理单帧
        try:
            stitched_image = visualizer.process_single_frame(point_cloud_path, image_paths, voxel_image_path, ts)
            image_with_tp = visualizer.draw_timestamp_on_image(stitched_image, tp_diff, cams_tp)
            # 保存结果
            output_path = f"{output_dir}/{ts}.png"
            cv2.imwrite(output_path, stitched_image)
            
        except Exception as e:
            print(f"Error processing frame {ts}: {e}")
            traceback.print_exc()
    
    # 生成视频
    print("Generating video...")
    video_save_path = f"{output_dir}/output_video.mp4"
    generate_video_from_frames(output_dir, video_save_path, fps=5)


if __name__ == "__main__":
    main("/home/jszr/Data/OCC_Autolabel/DATA/tank")

    # generate_video_from_frames("/home/jszr/Data/OCC_Autolabel/DATA/tank/video", "/home/jszr/Data/OCC_Autolabel/DATA/tank/video/output_video.mp4", fps=5)