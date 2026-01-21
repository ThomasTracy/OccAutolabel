import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .occ_autolabel_base import OCCAutolabelBase


class OCCAutolabelwithUEPose(OCCAutolabelBase):
    """
    Input is lidar points with semantics
    pose is from UE not from mujuco
    pose is save in first 2 points of lidar point data
    """
    def __init__(self, config):
        super(OCCAutolabelwithUEPose, self).__init__(config)

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
        UE中的pose是和lidar的timestamp一致的
        """
        return self.poses[timestamp]

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

        r = R.from_euler('xyz', angular, degrees=True)
        rotation = r.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array(translation)
        return transform

    def process_multi_frame(self):
        for i, timestamp in enumerate(self.timestamps):
            frame_radius = self.config['multi_frame_radius']
            selected_points = []
            selected_semantics = []
            full_points_in_world = []
            full_semantics = []
            ref_pose = self.get_pose(timestamp)
            ref_position = np.array(ref_pose['translation'])
            ref_angular = np.array(ref_pose['rotation'])
            ref_to_world = self.trans_angular_to_matrix4x4(ref_position, ref_angular)

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
                cur_pos = np.array(cur_pose['translation'])
                cur_angular = np.array(cur_pose['rotation'])

                cur_to_world = self.trans_angular_to_matrix4x4(cur_pos, cur_angular)

                lidar_points = self.load_lidar_pointcloud(tp, self.config['point_dim'])
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
                # UE的pose为lidar传感器的pose，此处直接将lidar坐标系转换到世界坐标系中
                points_in_world = lidar_points_homo @ cur_to_world.T

                # 高度范围在世界坐标系中截取
                mask2 = (points_in_world[:, 2] > self.config['pc_range'][2]) & (points_in_world[:, 2] < self.config['pc_range'][5])

                # UE pose 为lidar的pose，此处还要乘以lidar到ego的变换矩阵
                points_on_ref_lidar = points_in_world @ np.linalg.inv(ref_to_world).T
                points_on_ref = points_on_ref_lidar @ lidar_to_ego

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
            # input("finish saving world points......")
            if self.config['debug_mode']:
                print("------------ debug Mode ------------")
                return

            self.process_single_frame(timestamp, selected_points)
            self.logger.info("------------ finish processing frame {} ------------".format(timestamp))

