import os
import numpy as np

from .occ_autolabel_base import OCCAutolabelBase


class OCCAutolabelwithLidarSegmentation(OCCAutolabelBase):
    """
    Input is lidar points with semantics
    """
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
            lidar_points = self.load_lidar_pointcloud(timestamp, self.config['point_dim'])[2:]

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
            cur_to_world = self.projector.to_matrix4x4(current_quaternion, current_position)
            
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
                start_idx = max(0,len(self.timestamps) - 2 * frame_radius)
            selected_tp = self.timestamps[start_idx:end_idx]
            # selected_tp = random.sample(self.timestamps, 100)
            
            # 取当前帧的前 frame_num 帧进行拼接
            for tp in selected_tp:
                cur_pose = self.get_pose(tp)
                current_position = np.array(cur_pose['translation'])
                current_position[0] += np.random.uniform(-0.2, 0.2)
                current_position[1] += np.random.uniform(-0.2, 0.2)
                current_quaternion = np.array(cur_pose['rotation'])
                cur_to_world = self.projector.to_matrix4x4(current_quaternion, current_position)

                lidar_points = self.load_lidar_pointcloud(tp,self.config['point_dim'])[2:]
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
                if 'ego_range' in self.config:
                    mask = (points_on_ego[:, 0] > self.config['ego_range'][2]) | (points_on_ego[:, 0] < self.config['ego_range'][0]) | \
                            (points_on_ego[:, 1] > self.config['ego_range'][3]) | (points_on_ego[:, 1] < self.config['ego_range'][1])
                else:
                    mask = (points_on_ego[:, 0] > 0.5) | (points_on_ego[:, 0] < -0.5) | \
                            (points_on_ego[:, 1] > 0.5) | (points_on_ego[:, 1] < -0.5)

                # ego --> world --> reference frame
                points_in_world = points_on_ego @ cur_to_world.T

                # 高度范围z在世界坐标系中截取
                if 'voxel_range' in self.config:
                    mask2 = (points_in_world[:, 2] > self.config['voxel_range'][2]) & (points_in_world[:, 2] < self.config['voxel_range'][5])
                    mask = mask & mask2

                # 点云从世界坐标系转换到参考帧所在坐标系
                points_on_ref = points_in_world @ np.linalg.inv(ref_to_world).T

                # x,y 范围只保留voxel_range范围内的点
                if 'voxel_range' in self.config:
                    mask3 = (points_on_ref[:, 0] > self.config['voxel_range'][0]) & (points_on_ref[:, 0] < self.config['voxel_range'][3]) & \
                            (points_on_ref[:, 1] > self.config['voxel_range'][1]) & (points_on_ref[:, 1] < self.config['voxel_range'][4])
                
                    mask = mask & mask3

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
            input("Press Enter to continue ~~~~~~~")

            self.process_single_frame(timestamp, selected_points)
            self.logger.info("------------ finish processing frame {}, points {}".format(timestamp,selected_points.shape[0]))
            if debug_timestamp is not None:
                return