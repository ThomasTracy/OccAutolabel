import numpy as np
import cv2
from pyquaternion import Quaternion


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
    
    def multi_lidar_concat(self, multi_frame_points, poses):
        '''
        将多个激光点云拼接成单个点云
        拼接到第一帧中
        '''
        pass

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
            points_2d = points_on_image[valid_mask][:,:2]
        return points_2d, valid_mask


    def visualize(self, points_2d, original_points=None):
        pass