import numpy as np
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


class Projector:
    def __init__(self,
                 image_shape=None,
                 camera_matrix=None,
                 distortion_coeffs=None,
                 camera_to_ego_rotation=None,
                 camera_to_ego_translation=None,
                 lidar_to_ego_rotation=None,
                 lidar_to_ego_translation=None):
        self.image_shape = image_shape
        if camera_matrix is None:
            self.camera_matrix = np.eye(3)
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        if camera_to_ego_rotation is None:
            camera_to_ego_rotation = np.eye(3)
        if camera_to_ego_translation is None:
            camera_to_ego_translation = np.zeros(3)
        if lidar_to_ego_rotation is None:
            lidar_to_ego_rotation = np.eye(3)
        if lidar_to_ego_translation is None:
            lidar_to_ego_translation = np.zeros(3)
        self.camera_to_ego_rotation = camera_to_ego_rotation
        self.camera_to_ego_translation = camera_to_ego_translation
        self.lidar_to_ego_rotation = lidar_to_ego_rotation
        self.lidar_to_ego_translation = lidar_to_ego_translation

        self.camera_to_ego = self.to_matrix4x4(camera_to_ego_rotation, camera_to_ego_translation)
        self.lidar_to_ego = self.to_matrix4x4(lidar_to_ego_rotation, lidar_to_ego_translation)
        self.ego_to_camera = np.linalg.inv(self.camera_to_ego)
        self.ego_to_lidar = np.linalg.inv(self.lidar_to_ego)
    
    def to_matrix4x4(self, rotation, translation=None):
        '''
        将旋转平移矩阵转换成4x4矩阵
        '''
        if rotation.shape != (3, 3):
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

    def cam_coord_to_image_coord(self, points):
        '''
        将点云从相机坐标系转换到图像坐标系
        '''
        points_xyz = points[:, :3]
        if self.distortion_coeffs is not None:
            points_on_image = cv2.projectPoints(
                points_xyz,
                np.eye(3),
                np.zeros((3, 1)),
                self.camera_matrix,
                self.distortion_coeff
            )
        else:
            points_on_image = (self.camera_matrix @ points_xyz.T).T     # 通过相机内参将点云投影到成像平面
        points_on_image[:,0] /= (points_on_image[:,2] + 1e-6)           # 通过深度缩放投影到像素平面
        points_on_image[:,1] /= (points_on_image[:,2] + 1e-6)

        # 筛选有效点，在相机前方，在图像内
        if self.image_shape is not None:
            valid_mask = (points_xyz[:,2] > 0) & \
                        (points_on_image[:,0] > 0) & (points_on_image[:,0] < self.image_shape[0]) & \
                        (points_on_image[:,1] > 0) & (points_on_image[:,1] < self.image_shape[1])
            points_2d = points_on_image[valid_mask][:,:2]
        return points_2d, valid_mask, points_on_image

    def lidar_coord_to_cam_coord(self, points):
        '''
        将点云从激光坐标系转换到相机坐标系
        '''
        transform = self.ego_to_camera @ self.lidar_to_ego
        points_on_cam_coord = points @ transform.T
        return points_on_cam_coord
    
    def multi_lidar_concat_org(self, multi_frame_points, poses, ref_pose):
        '''
        将多个激光点云拼接成单个点云
        拼接到第一帧中
        '''
        """
        拼接多帧点云到关键帧
        直接使用相对变换公式，不构建完整的变换矩阵
        """
        # 加载里程计数据
        
        # 获取参考帧的位姿
        ref_position = np.array(ref_pose['translation'])
        ref_quaternion = np.array(ref_pose['rotation'])
        ref_rotation = R.from_quat(ref_quaternion).as_matrix()
        
        merged_points = []
        
        for i, points in enumerate(multi_frame_points):
            if len(points) == 0:
                continue
            
            # 获取当前帧的位姿
            current_pose = poses[i]
            current_position = np.array(current_pose['translation'])
            current_quaternion = np.array(current_pose['rotation'])
            current_rotation = R.from_quat(current_quaternion).as_matrix()
            
            # 直接计算相对变换（当前坐标系 → 参考坐标系）
            # 旋转部分: R_current_to_ref = R_ref @ R_current^T
            R_current_to_ref = ref_rotation.T @ current_rotation
            
            # 平移部分: t_current_to_ref = R_ref @ (t_ref - t_current)
            relative_translation = ref_rotation.T @ (current_position - ref_position)
            
            # 直接应用变换到点云
            # p_ref = R_current_to_ref @ p_current + t_current_to_ref
            points_ref = (R_current_to_ref @ points.T).T + relative_translation
            
            # 添加到合并点云
            merged_points.append(points_ref)

        merged_points = np.vstack(merged_points)

        return merged_points
    
    def multi_lidar_concat(self, multi_frame_points, poses, ref_pose):
        '''
        将多个激光点云拼接成单个点云
        拼接到第一帧中
        '''
        # 获取参考帧的位姿
        ref_position = np.array(ref_pose['translation'])
        ref_quaternion = np.array(ref_pose['rotation'])
        # ref_rotation = R.from_quat(ref_quaternion).as_matrix()
        ref_to_world = self.to_matrix4x4(ref_quaternion, ref_position)
        
        merged_points = []
        points_in_world = []
        
        for i, points in enumerate(multi_frame_points):
            if len(points) == 0:
                continue
            
            # 获取当前帧的位姿
            current_pose = poses[i]
            current_position = np.array(current_pose['translation'])
            current_quaternion = np.array(current_pose['rotation'])
            # current_rotation = R.from_quat(current_quaternion).as_matrix()
            cur_to_world = self.to_matrix4x4(current_quaternion, current_position)
            
            points_homo = self.to_homo_coord(points)
            transform_matrix = np.linalg.inv(ref_to_world) @ cur_to_world
            points_transformed = (points_homo @ transform_matrix.T)[:, :3]
            # 添加到合并点云
            merged_points.append(points_transformed)

            p_in_world = (points_homo @ cur_to_world.T)[:, :3]
            points_in_world.append(p_in_world)

        merged_points = np.vstack(merged_points)

        points_in_world = np.vstack(points_in_world)
        # np.save(f"points_in_world.npy", points_in_world)
        # input("press any key to continue...")

        return merged_points


    def project_points_to_image(self, points_3d):
        points_3d = points_3d[:,:3]
        points_homo = self.to_homo_coord(points_3d)
        points_on_cam_coord = self.lidar_coord_to_cam_coord(points_homo)
        points_2d, valid_mask, full_points2d_with_depth = self.cam_coord_to_image_coord(points_on_cam_coord)
        return points_2d, valid_mask, full_points2d_with_depth
    
    def projection(self, points_3d, 
                    image_shape=None,
                    camera_matrix=None,
                    distortion_coeffs=None,
                    camera_to_ego_rotation=None,
                    camera_to_ego_translation=None,
                    lidar_to_ego_rotation=None,
                    lidar_to_ego_translation=None):
        points_3d = points_3d[:,:3]
        points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        if camera_to_ego_rotation is None:
            camera_to_ego_rotation = np.eye(3)
        if camera_to_ego_translation is None:
            camera_to_ego_translation = np.zeros(3)
        if lidar_to_ego_rotation is None:
            lidar_to_ego_rotation = np.eye(3)
        if lidar_to_ego_translation is None:
            lidar_to_ego_translation = np.zeros(3)

        camera_to_ego = self.to_matrix4x4(camera_to_ego_rotation, camera_to_ego_translation)
        lidar_to_ego = self.to_matrix4x4(lidar_to_ego_rotation, lidar_to_ego_translation)
        ego_to_camera = np.linalg.inv(camera_to_ego)
        ego_to_lidar = np.linalg.inv(lidar_to_ego)

        # lidar 坐标系到 camera 坐标系
        transform = ego_to_camera @ lidar_to_ego
        points_on_cam_coord = points_homo @ transform.T

        # camera 坐标系 到 image 坐标系
        points_xyz = points_on_cam_coord[:, :3]
        if distortion_coeffs is not None:
            points_on_image = cv2.projectPoints(
                points_xyz,
                np.eye(3),
                np.zeros((3, 1)),
                camera_matrix,
                distortion_coeffs
            )
        else:
            points_on_image = (camera_matrix @ points_xyz.T).T     # 通过相机内参将点云投影到成像平面
        points_on_image[:,0] /= (points_on_image[:,2] + 1e-6)           # 通过深度缩放投影到像素平面
        points_on_image[:,1] /= (points_on_image[:,2] + 1e-6)

        # 筛选有效点，在相机前方，在图像内
        if image_shape is not None:
            valid_mask = (points_xyz[:,2] > 0) & \
                        (points_on_image[:,0] > 0) & (points_on_image[:,0] < image_shape[0]) & \
                        (points_on_image[:,1] > 0) & (points_on_image[:,1] < image_shape[1])
            points_on_image = points_on_image[valid_mask]
        return points_on_image, valid_mask


    def visualize(self, points_2d, original_points=None):
        pass