import os
import logging
import numpy as np

from collections import deque, defaultdict
from cores.utils.projector import Projector

from cores.category import Category

from tools.crop_partial_points import bin_to_pcd

class Processor:
    def __init__(self, config):
        self.config = config
        self.projector = Projector()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)


class PreProcessor(Processor):
    def __init__(self, config):
        super().__init__(config)
    
    def preprocess_cloud(self, points):
        '''
        输入为点云
        
        :param points: List[[x,y,z,semantic],...]
        :return preprocessed_points: List[]
        '''
        pass
    

    def remove_ego_points(self, points):
        '''
        输入为点云
        对点云进行ego-point去除,去除属于自车的点
        顺便去除地面以下的点
        '''
        x_min,y_min, x_max, y_max = self.config['ego_range']
        mask = (points[:, 0] > x_max) | (points[:, 0] < x_min) | \
                (points[:, 1] > y_max) | (points[:, 1] < y_min)
        # mask = mask & (points[:, 2] > 0.0)
        points = points[mask]
        points = points[:,:3]

        return points

    
    def redefine_blind_floor_points(self, points):
        '''
        输入为点云
        将视角盲区中的地面点云重新赋值为地面语义
        
        :param points: List[[x,y,z,semantic],...]
        :return preprocessed_points: List[[x,y,z,semantic],...]
        '''
        x_min, y_min, z_min, x_max, y_max, z_max = self.config['blind_floor_range']
        # 获取坐标点
        coords = points[:, :3]  # 获取x, y, z坐标
        # 创建掩码，找出在指定范围内的点
        ground_mask = (
            (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
            (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
        )
        # 将范围内点的语义标签设为地面点
        points[ground_mask, 3] = Category.ROAD.value
        return points


class PostProcessor(Processor):
    def __init__(self, config):
        super().__init__(config)


    def points_in_range(self, points):
        '''
        输入为点云
        对点云进行裁剪
        
        :param points: List[[x,y,z,semantic],...]
        :return cropped_points: List[[x,y,z,semantic],...]
        '''
        x_min, y_min, z_min, x_max, y_max, z_max = self.config['voxel_range']
        valid_mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[valid_mask]

    
    def split_ground_points(self, points):
        '''
        输入为点云
        对点云进行地面分割
        输出为地面点云和剩余点云
        
        :param points: List[[x,y,z,semantic],...]
        :return ground_points: List[[x,y,z,semantic],...]
        :return non_ground_points: List[[x,y,z,semantic],...]
        '''
        ground_points = []
        non_ground_points = []
        
        # 遍历每一帧点云
        for frame_points in points:
            if frame_points.shape[1] < 4:
                raise ValueError("Point cloud must contain semantic information (4th column)")
            
            # 获取语义标签列
            semantic_labels = frame_points[:, 3]
            
            # 创建掩码
            ground_mask = (semantic_labels == Category.ROAD.value)
            non_ground_mask = (semantic_labels != Category.ROAD.value)
            
            # 分离点云
            frame_ground_points = frame_points[ground_mask]
            frame_non_ground_points = frame_points[non_ground_mask]
            
            # 添加到结果列表
            ground_points.append(frame_ground_points)
            non_ground_points.append(frame_non_ground_points)
        
        return ground_points, non_ground_points
    

    def split_dynamic_objects(self, points):
        '''
        输入为点云
        对点云进行动态物体分割，将语义标签为1和2的点作为一个列表，其余点作为另一个列表
        
        :param points: List[np.array(Nx4), np.array(Mx4), ...] - 每个数组包含[x,y,z,semantic]
        :return dynamic_points: List[np.array(Kx4), np.array(Lx4), ...] - 每帧的动态物体点(语义为1和2)
        :return static_points: List[np.array(Px4), np.array(Qx4), ...] - 每帧的静态物体点(语义不为1和2)
        '''
        dynamic_points = []
        static_points = []
        
        # 遍历每一帧点云
        for frame_points in points:
            if frame_points.shape[1] < 4:
                raise ValueError("Point cloud must contain semantic information (4th column)")
            
            # 获取语义标签列
            semantic_labels = frame_points[:, 3]
            
            # 创建掩码
            dynamic_mask = (semantic_labels == 1) | (semantic_labels == 2)
            static_mask = (semantic_labels != 1) & (semantic_labels != 2)
            
            # 分离点云
            frame_dynamic_points = frame_points[dynamic_mask]
            frame_static_points = frame_points[static_mask]
            
            # 添加到结果列表
            dynamic_points.append(frame_dynamic_points)
            static_points.append(frame_static_points)
        
        return dynamic_points, static_points


    def static_noise_filter(self, points):
        '''
        针对静态物体的点云处理，将散点当做噪声去除
        
        :param points: np.array(N x 4)
        '''
        pass

    def voxel_clustering_denoise(self, points):
        """
        体素化聚类去噪, points全部为静态物体的点
        
        参数:
            points: np.array, 形状为 (N, 4), 每行包含 [x, y, z, semantic]
            
        返回:
            filtered_points: np.array, 去噪后的点云，形状为 (M, 4)，M <= N
        """
        
        voxel_size = self.config['rough_voxel_size']
        threshold_point = self.config['valid_voxel_point_threshold']
        threshold_poly = self.config['valid_polygon_voxel_threshold']

        # 1. 体素化
        def compute_voxel_indices(points, voxel_size):
            """计算每个点所属的体素索引"""
            # 计算体素坐标
            voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
            return voxel_coords
        
        def get_voxel_key(voxel_coord):
            """将体素坐标转换为可哈希的元组"""
            return tuple(voxel_coord)
        
        # 计算体素索引
        voxel_coords = compute_voxel_indices(points, voxel_size)
        
        # 组织体素数据结构
        voxel_dict = {}  # key: 体素坐标, value: {points_indices, count, is_noise}
        
        for i, voxel_coord in enumerate(voxel_coords):
            voxel_key = get_voxel_key(voxel_coord)
            
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points_indices': [],
                    'count': 0,
                    'is_noise': False,
                    'visited': False,
                    'coord': voxel_coord
                }
            
            voxel_dict[voxel_key]['points_indices'].append(i)
            voxel_dict[voxel_key]['count'] += 1
        
        # 2. 根据点数标记噪声体素
        for voxel_key in voxel_dict:
            if voxel_dict[voxel_key]['count'] < threshold_point:
                voxel_dict[voxel_key]['is_noise'] = True
        
        # 3. 26邻域区域增长聚类
        def get_26_neighbors(voxel_coord):
            """获取26邻域的体素坐标"""
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # 跳过自身
                        neighbor_coord = voxel_coord + np.array([dx, dy, dz])
                        neighbors.append(tuple(neighbor_coord))
            return neighbors
        
        clusters = []  # 存储聚类结果，每个聚类是体素坐标的列表
        
        # 遍历所有非噪声且未访问的体素
        for voxel_key, voxel_data in voxel_dict.items():
            if voxel_data['is_noise'] or voxel_data['visited']:
                continue
            
            # 开始新的聚类
            cluster = []
            queue = deque([voxel_key])
            voxel_data['visited'] = True
            
            while queue:
                current_key = queue.popleft()
                current_data = voxel_dict[current_key]
                
                # 只将非噪声体素加入聚类
                if not current_data['is_noise']:
                    cluster.append(current_key)
                
                # 获取26邻域
                neighbors = get_26_neighbors(current_data['coord'])
                
                # 检查每个邻域体素
                for neighbor_key in neighbors:
                    if neighbor_key in voxel_dict:
                        neighbor_data = voxel_dict[neighbor_key]
                        
                        # 如果邻居未访问且不是噪声（或者即使是噪声也需要检查连通性）
                        if not neighbor_data['visited']:
                            neighbor_data['visited'] = True
                            # 只有非噪声体素才加入队列继续扩展
                            if not neighbor_data['is_noise']:
                                queue.append(neighbor_key)
            
            if cluster:  # 只添加非空的聚类
                clusters.append(cluster)
        
        # 4. 根据聚类大小标记噪声
        # 创建一个映射：体素 -> 所属聚类索引
        voxel_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters):
            for voxel_key in cluster:
                voxel_to_cluster[voxel_key] = cluster_idx
        
        # 标记小聚类中的体素为噪声
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) < threshold_poly:
                for voxel_key in cluster:
                    if voxel_key in voxel_dict:
                        voxel_dict[voxel_key]['is_noise'] = True
        
        # 5. 收集非噪声点
        filtered_indices = []
        noise_indices = []
        
        # 先收集所有非噪声体素中的点
        for voxel_key, voxel_data in voxel_dict.items():
            if not voxel_data['is_noise']:
                filtered_indices.extend(voxel_data['points_indices'])
            else:
                noise_indices.extend(voxel_data['points_indices'])
        
        # 如果没有找到任何非噪声点，返回空数组
        if not filtered_indices:
            return np.array([], dtype=points.dtype).reshape(0, 4)
        
        # 获取去噪后的点
        filtered_points = points[filtered_indices]
        noise_points = points[noise_indices]
        
        return filtered_points
    

    def bev_mask_ground_denoise(self, points):
        """
        使用地面点生成BEV网格,并过滤静态点中的噪声
        
        逻辑：
        1. 用ground点在BEV平面生成2D高度图
        2. 计算static point与该网格ground中最高ground点的相对高度差
        3. 如果相对高度差小于阈值，认定为噪声删除
        
        参数:
        points: 输入点云，形状为 (N, 4)，每行包含 [x, y, z, semantic]
            semantic: 0-地面点, 1-静态点
        
        返回:
        filtered_points: 过滤后的点云
        """
        grid_size = self.config['ground_mask_grid_size']
        height_threshold = self.config['ground_height_threshold']  # 相对高度阈值

        # 分离地面点和静态点
        ground_mask = points[:, 3] == Category.ROAD.value
        static_mask = points[:, 3] == Category.STATIC_OBJECT.value
        
        ground_points = points[ground_mask]
        static_points = points[static_mask]
        other_points = points[~(ground_mask | static_mask)]  # 保留其他类型点
        
        if len(ground_points) == 0:
            print("警告：未找到地面点，跳过噪声过滤")
            return points
        
        # 1. 计算BEV平面的边界范围（使用地面点）
        x_min, x_max = ground_points[:, 0].min(), ground_points[:, 0].max()
        y_min, y_max = ground_points[:, 1].min(), ground_points[:, 1].max()
        
        # 2. 创建网格索引计算函数
        def compute_grid_indices(points_xy, x_min, y_min, grid_size, x_bins, y_bins):
            """向量化计算网格索引"""
            x_indices = np.floor((points_xy[:, 0] - x_min) / grid_size).astype(int)
            y_indices = np.floor((points_xy[:, 1] - y_min) / grid_size).astype(int)
            
            # 限制索引范围
            x_indices = np.clip(x_indices, 0, x_bins - 1)
            y_indices = np.clip(y_indices, 0, y_bins - 1)
            
            return x_indices, y_indices
        
        # 3. 计算网格行列数
        eps = 1e-6
        x_bins = int(np.ceil((x_max - x_min + eps) / grid_size))
        y_bins = int(np.ceil((y_max - y_min + eps) / grid_size))
        
        # 4. 生成地面高度图（关键改进：保存每个网格的ground最高点高度）
        ground_x_idx, ground_y_idx = compute_grid_indices(
            ground_points[:, :2], x_min, y_min, grid_size, x_bins, y_bins
        )
        
        # 使用字典记录每个网格的ground点最高高度
        from collections import defaultdict
        grid_z_max = defaultdict(lambda: -np.inf)
        
        for i in range(len(ground_points)):
            grid_key = (ground_x_idx[i], ground_y_idx[i])
            grid_z_max[grid_key] = max(grid_z_max[grid_key], ground_points[i, 2])
        
        # 创建ground高度图（使用最高点高度）
        ground_height_grid = np.full((x_bins, y_bins), np.nan, dtype=np.float32)
        for (gx, gy), z_max in grid_z_max.items():
            ground_height_grid[gx, gy] = z_max
        
        # 5. 过滤静态点（使用相对高度）
        if len(static_points) > 0:
            # 计算静态点的网格索引
            static_x_idx, static_y_idx = compute_grid_indices(
                static_points[:, :2], x_min, y_min, grid_size, x_bins, y_bins
            )
            
            # 获取每个静态点所在网格的ground高度
            ground_heights_at_static = ground_height_grid[static_x_idx, static_y_idx]
            
            # 条件1: 该网格有ground点（ground_height不是nan）
            has_ground_mask = ~np.isnan(ground_heights_at_static)
            
            # 条件2: 计算相对高度差（static_z - ground_z）
            relative_height = static_points[:, 2] - ground_heights_at_static
            
            # 条件3: 相对高度差小于阈值（太靠近ground，认为是噪声）
            too_close_mask = relative_height < height_threshold
            
            # 同时满足：有ground且太接近 → 噪声
            noise_mask = has_ground_mask & too_close_mask
            
            # 保留不是噪声的静态点
            keep_static_mask = ~noise_mask
            filtered_static_points = static_points[keep_static_mask]
            
            # removed_count = np.sum(noise_mask)
            
            # 调试信息
            # if removed_count > 0:
                # print(f"[Ground去噪] 移除 {removed_count}/{len(static_points)} 个静态噪声点")
        else:
            filtered_static_points = static_points
        
        # 6. 合并所有点
        if len(other_points) > 0:
            filtered_points = np.concatenate([ground_points, filtered_static_points, other_points], axis=0)
        else:
            filtered_points = np.concatenate([ground_points, filtered_static_points], axis=0)
        
        return filtered_points


    def ground_denoise_by_std_deviation(self, points):
        """
        基于标准差的方法筛选出高于或者低于地面很多的离群点，将其删除
        
        参数：
        - points: (N, 3) 点云数组, 第三列为Z值
        - std_multiplier: 标准差倍数阈值, 默认3
        
        返回：
        - filtered_points: 滤除后的点云
        """
        std_multiplier = self.config['std_multiplier']

        z_values = points[:, 2]
        
        # 计算统计量
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        
        # 计算阈值
        lower_threshold = z_mean - std_multiplier * z_std
        upper_threshold = z_mean + std_multiplier * z_std
        
        # 滤除离群点
        mask = (z_values >= lower_threshold) & (z_values <= upper_threshold)
        filtered_points = points[mask]
        
        return filtered_points
    

    def dynamic_points_denoise(self, point_cloud):
        """
        对动态物体附近的静态点进行滤除
        """

        assert point_cloud.shape[1] == 4

        voxel_size = self.config['dynamic_denoise_voxel_size']
        dilate_radius = self.config['dynamic_denoise_dilate_radius']
        static_label = Category.STATIC_OBJECT.value,
        dynamic_label = [Category.PEOPLE.value, Category.VEHICLE.value]
        ground_z_thresh = self.config['dynamic_denoise_ground_z_thresh']

        xyz = point_cloud[:, :3]
        semantic = point_cloud[:, 3].astype(np.int32)

        # ------------------------------------------------
        # 1. voxelization
        # ------------------------------------------------
        voxel_idx = np.floor(xyz / voxel_size).astype(np.int32)

        voxel_map = defaultdict(list)
        for i, v in enumerate(voxel_idx):
            voxel_map[tuple(v)].append(i)

        # ------------------------------------------------
        # 2. collect dynamic voxels
        # ------------------------------------------------
        dynamic_voxels = set()
        for i, s in enumerate(semantic):
            if s in dynamic_label:
                dynamic_voxels.add(tuple(voxel_idx[i]))

        if len(dynamic_voxels) == 0:
            return point_cloud.copy()

        # ------------------------------------------------
        # 3. voxel dilation (neighbor propagation)
        # ------------------------------------------------
        expanded_voxels = set()
        r = dilate_radius

        for vx, vy, vz in dynamic_voxels:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        # 球形邻域（比立方体更保守）
                        if dx*dx + dy*dy + dz*dz <= r*r:
                            expanded_voxels.add(
                                (vx + dx, vy + dy, vz + dz)
                            )

        # ------------------------------------------------
        # 4. mark noise points (to be removed)
        # ------------------------------------------------
        remove_mask = np.zeros(len(point_cloud), dtype=bool)

        for voxel in expanded_voxels:
            if voxel not in voxel_map:
                continue

            for idx in voxel_map[voxel]:
                # 只处理 static
                if semantic[idx] != static_label:
                    continue

                # -------------------------------
                # 地面保护逻辑（核心）
                # -------------------------------
                # 1) semantic 是 ground → 不删
                # if semantic[idx] == ground_label:
                #     continue

                # 2) z 很低的点认为是地面（即使被误标为 static）
                if xyz[idx, 2] < ground_z_thresh:
                    continue

                # 满足条件 → 删除
                remove_mask[idx] = True

        # ------------------------------------------------
        # 5. remove noise points
        # ------------------------------------------------
        filtered_pc = point_cloud[~remove_mask]

        return filtered_pc
    

    def assign_unknown_below_ground(
        self,
        voxel_grid,
        radius=2,
        max_height=1
    ):
        """
        填充低于地面的未知点

        voxel_grid: np.ndarray, shape (X, Y, Z)
        voxel_grid[x, y, z] = semantic
        radius: 邻域填充的搜索半径
        max_height: 最大高度阈值，超过该高度的不进行处理，单位为voxel
        """
        res_voxel_grid = voxel_grid.copy()

        semantic_static_with_unknown = [
            Category.STATIC_OBJECT.value,
            Category.ROAD.value,
            Category.UNKNOWN.value
        ]

        semantic_unknown =  Category.UNKNOWN.value

        X, Y, Z = voxel_grid.shape
        z_indices = range(max_height)

        # 记录哪些 (x, y) 没有在 Rule 1 中找到 ground
        no_ground_columns = []

        # -------------------------
        # Rule 1: 同柱 z 方向
        # -------------------------
        for x in range(X):
            for y in range(Y):
                ground_z = None

                for z in z_indices:
                    if voxel_grid[x, y, z] in semantic_static_with_unknown:
                        ground_z = z
                        break

                if ground_z is not None:
                    # ground 以下全部置 unknown
                    if ground_z > 0:
                        res_voxel_grid[x, y, :ground_z] = semantic_unknown
                else:
                    no_ground_columns.append((x, y))

        # -------------------------
        # Rule 2: 邻域补偿
        # -------------------------
        # 预计算邻域 offset
        offsets = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    offsets.append((dx, dy))

        for x, y in no_ground_columns:
            est_ground_z = None
            for z in z_indices:
                if est_ground_z is not None:
                    break
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < X and 0 <= ny < Y:
                        if voxel_grid[nx, ny, z] in semantic_static_with_unknown:
                            est_ground_z = z
                            break  # 只取该邻柱最低的非 free

            if est_ground_z is not None and est_ground_z > 0:
                res_voxel_grid[x, y, :est_ground_z] = semantic_unknown

        return res_voxel_grid


    def raycast_3d(self, origin, end_voxel, voxel_grid, max_steps=10000):
        
        # 只通过非地面的静态物体进行遮挡判断
        semantic_static_nonground = [
            Category.STATIC_OBJECT.value
        ]
        semantic_dynamic = [
            Category.VEHICLE.value,
            Category.PEOPLE.value
        ]

        x, y, z = origin.astype(int)
        ex, ey, ez = end_voxel.astype(int)

        dx, dy, dz = ex - x, ey - y, ez - z
        step_x, step_y, step_z = np.sign(dx), np.sign(dy), np.sign(dz)

        t_delta_x = abs(1.0 / dx) if dx != 0 else np.inf
        t_delta_y = abs(1.0 / dy) if dy != 0 else np.inf
        t_delta_z = abs(1.0 / dz) if dz != 0 else np.inf

        def intbound(s, ds):
            if ds > 0:
                return (np.floor(s + 1) - s) / ds
            elif ds < 0:
                return (s - np.floor(s)) / -ds
            return np.inf

        t_max_x = intbound(x, dx)
        t_max_y = intbound(y, dy)
        t_max_z = intbound(z, dz)

        # 记录当前射线的第一个hit点，无论是终点点还是终点前的点
        first_hit_x, first_hit_y, first_hit_z = -1, -1, -1

        for _ in range(max_steps):

            # ✅ 边界保护（最重要）
            if not (0 <= x < voxel_grid.shape[0] and
                    0 <= y < voxel_grid.shape[1] and
                    0 <= z < voxel_grid.shape[2]):
                break

            sem = voxel_grid[x, y, z]

            # ---------- STATIC ----------
            if sem in semantic_static_nonground:
                # 紧邻的voxel会存在遮挡，这种遮挡不算，必须保证超过1个voxel距离
                if abs(x - ex) > 1 or abs(y - ey) > 1 or abs(z - ez) > 1:
                    first_hit_x, first_hit_y, first_hit_z = x, y, z
                    break
            
            # 到达终点时
            if (x, y, z) == (ex, ey, ez):
                # hit_end = True
                first_hit_x, first_hit_y, first_hit_z = x, y, z
                break

            # ---------- dynamic / free ----------
            # 默认 FREE，不用写

            # ---------- DDA ----------
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        # ---------- static 后 → UNKNOWN ----------
        while True:
            # 先迭代下一步，否则会造成终点点被改为 UNKNOWN的问题
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

            if not (0 <= x < voxel_grid.shape[0] and
                    0 <= y < voxel_grid.shape[1] and
                    0 <= z < voxel_grid.shape[2]):
                break
            
            # 应该是多余的一步
            if first_hit_x == -1 or first_hit_y == -1 or first_hit_z == -1:
                break

            # 当前点若紧邻第一次hit的点，则跳过，避免错误遮挡
            if abs(x - first_hit_x) < 2 and abs(y - first_hit_y) < 2 and abs(z - first_hit_z) < 2:
                continue
            
            # 如果碰到了 dynamic，则跳过，后续也不处理
            if voxel_grid[x, y, z] in semantic_dynamic:
                break
            
            # 只对静态物体后面遮挡的 静态物体 进行 unknown 标记
            # 没有占据的点永远是 free，不用改
            if voxel_grid[x, y, z] == Category.STATIC_OBJECT.value:
                voxel_grid[x, y, z] = Category.UNKNOWN.value

    def calculate_free_unknown(self, points):

        '''
        给占据栅格添加 区分free 和 unknown 语义
        通过ray casting 实现
        
        points: np.ndarray [N, 4], x,y,z,semantic, 栅格的中心点和语义
        '''

        semantic_occupied = [
            Category.ROAD.value,
            Category.STATIC_OBJECT.value,
            Category.VEHICLE.value,
            Category.PEOPLE.value
        ]

        voxel_size = self.config['voxel_size']
        sensor_origin = self.config['sensor_origin']

        below_ground_max_height = 0.5
        xyz = points[:, :3]
        sem = points[:, 3].astype(int)

        vox = np.floor(xyz / voxel_size).astype(int)
        vox = np.column_stack([vox, sem])

        min_v = vox[:, :3].min(0)
        max_v = vox[:, :3].max(0)
        size = max_v - min_v + 1

        # 1. 全部初始化为 FREE
        voxel_grid = np.full(size, Category.FREE.value, np.int32)

        # 2. 只对 occupied voxel raycast
        occupied_mask = np.isin(
            sem,
            [Category.ROAD.value,
            Category.STATIC_OBJECT.value,
            Category.VEHICLE.value,
            Category.PEOPLE.value]
        )

        # 3. 写入 occupied 语义
        for x, y, z, s in vox[occupied_mask]:
            voxel_grid[x-min_v[0], y-min_v[1], z-min_v[2]] = s

        # 4. sensor origin
        origin_vox = np.floor(np.array(sensor_origin) / voxel_size).astype(int)
        origin_vox -= min_v

        # 👉 静态优先（推荐）
        static_mask = np.isin(sem, [Category.ROAD.value, Category.STATIC_OBJECT.value])
        static_mask_height_range = (points[:,2] > 0.5) & static_mask

        for v in vox[static_mask_height_range]:
            # 终点不在 occupied 语义中，之前被更改过了，则不处理
            v[:3] = v[:3] - min_v
            if voxel_grid[v[0],v[1],v[2]] not in semantic_occupied:
                continue
            self.raycast_3d(origin_vox, v[:3], voxel_grid)

        # 5. recover to world
        xs, ys, zs = np.meshgrid(
            np.arange(size[0]),
            np.arange(size[1]),
            np.arange(size[2]),
            indexing='ij'
        )

        # max_height_voxel = int(below_ground_max_height / voxel_size) - min_v[2]
        # voxel_grid = self.assign_unknown_below_ground(voxel_grid, radius=3,
        #                                         max_height=max_height_voxel)

        coords = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]) + min_v
        world_xyz = (coords + 0.5) * voxel_size
        sem_out = voxel_grid.ravel()

        return np.column_stack([world_xyz, sem_out])

    def full_point_process(self, points):
        '''
        输入为带有语义的点云, 全量拼接点云的处理
        对其进行点云处理，包括聚类，动态物体补全，去噪等
        输出为处理后的点云
        
        :param points: List[[x,y,z,semantic],...]
        :return postprocessed_points: List[[x,y,z,semantic],...]
        '''
        # ----------------------------------- 地面点和非地面点分离 -----------------------------------
        ground_points_list, non_ground_points_list = self.split_ground_points(points)

        # ----------------------------------- 动态物体和静态物体分离 -----------------------------------
        dynamic_points_list, static_points_list = self.split_dynamic_objects(non_ground_points_list)

        static_points = np.vstack(static_points_list)
        ground_points = np.vstack(ground_points_list)        

        # 聚类去噪移到了post_process_on_ego中
        # filtered_static_points = self.voxel_clustering_denoise(static_points)
        filtered_static_points = static_points

        # ----------------------------------- 合并地面点和静态物体点 -----------------------------------
        non_dynamic_points = np.vstack([filtered_static_points, ground_points])

        return non_dynamic_points
    

    def refine_ground_by_plane_fitting(self, points):
        """
        使用RANSAC平面拟合对ground和static点进行重新分类（优化版）
        
        参数:
            points: np.array (N, 4), 点云数据 [x, y, z, semantic]
            
        返回:
            refined_ground: np.array, 重新分类后的地面点
            refined_static: np.array, 重新分类后的静态点
        """
        # messi 动态点和静态点地面去噪
        non_ground_points = points[points[:, 3] != Category.ROAD.value]
        ground_points = points[points[:, 3] == Category.ROAD.value]
        
        if len(ground_points) == 0:
            print("警告：地面点为空，跳过平面拟合")
            return points
        
        if len(non_ground_points) == 0:
            print("警告：静态点为空，跳过重分类")
            return points
        
        # 获取配置参数
        height_threshold = self.config['ground_plane_height_threshold']
        
        # 🚀 优化1: 降采样ground点以加速RANSAC
        ground_xyz = ground_points[:, :3]
        n_points = len(ground_xyz)
        
        # 如果点数过多，随机采样以加速
        max_sample_points = 10000  # 最多使用10000个点进行拟合
        if n_points > max_sample_points:
            sample_indices = np.random.choice(n_points, max_sample_points, replace=False)
            ground_xyz_sampled = ground_xyz[sample_indices]
            # print(f"降采样：{n_points} -> {max_sample_points} 个ground点用于RANSAC")
        else:
            ground_xyz_sampled = ground_xyz
        
        # RANSAC 平面拟合参数
        max_iterations = 200  # 🚀 优化2: 降低迭代次数（原1000->200）
        sample_size = 3
        distance_threshold = height_threshold
        min_inliers_ratio = 0.3
        
        # -------------------- RANSAC 平面拟合 --------------------
        best_plane = None
        best_inliers_count = 0
        n_sampled = len(ground_xyz_sampled)
        
        if n_sampled < sample_size:
            print(f"警告：地面点数量({n_sampled})不足，无法拟合平面")
            return points
        
        # 🚀 优化3: 预分配数组避免重复创建
        for iteration in range(max_iterations):
            # 1. 随机选择3个点
            sample_indices = np.random.choice(n_sampled, sample_size, replace=False)
            p1, p2, p3 = ground_xyz_sampled[sample_indices]
            
            # 2. 计算法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            # 检查退化
            normal_length = np.linalg.norm(normal)
            if normal_length < 1e-6:
                continue
            
            normal = normal / normal_length
            d = -np.dot(normal, p1)
            
            # 3. 向量化计算距离
            distances = np.abs(np.dot(ground_xyz_sampled, normal) + d)
            
            # 4. 统计内点
            inliers_count = np.sum(distances < distance_threshold)
            
            # 5. 更新最佳模型
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_plane = (normal.copy(), d)  # 注意复制
        
        # 检查是否找到有效平面
        if best_plane is None or best_inliers_count < n_sampled * min_inliers_ratio:
            print(f"警告：RANSAC未找到有效地面平面（内点数: {best_inliers_count}/{n_sampled}）")
            return points
        
        normal, d = best_plane
        
        # 🚀 优化4: 移除SVD优化步骤（主要性能瓶颈）
        # 直接使用RANSAC得到的平面，精度已足够
        # SVD对大规模点云性能影响极大，可以跳过
        
        # print(f"✅ 平面拟合成功：法向量={normal}, d={d:.3f}, 内点数={best_inliers_count}/{n_sampled}")
        
        # -------------------- 重新分类 static 点 --------------------
        if len(non_ground_points) > 0:
            non_ground_xyz = non_ground_points[:, :3]
            
            # 向量化计算距离
            abs_distances = np.abs(np.dot(non_ground_xyz, normal) + d)
            
            # 创建掩码
            to_ground_mask = abs_distances < height_threshold
            
            # 🚀 优化5: 直接修改semantic值，避免数组复制
            # 创建结果数组副本
            result_points = points.copy()
            
            # 找到static点在原始数组中的位置
            static_mask = result_points[:, 3] != Category.ROAD.value
            static_indices = np.where(static_mask)[0]
            
            # 将需要重分类的点的semantic改为ground
            reclassified_indices = static_indices[to_ground_mask]
            result_points[reclassified_indices, 3] = Category.ROAD.value
            
            reclassified_count = len(reclassified_indices)
            # print(f"✅ 重新分类：{reclassified_count}/{len(static_points)} 个static点归为ground")
            
            return result_points
        else:
            return points


    def filter_dynamic_objects(self, points):
        """
        通过体素化和聚类算法，滤除动态物体点
        
        参数:
            points: np.array (N, 4), 不含地面点的点云 [x, y, z, semantic]
                    只包含静态物体(Category.STATIC_OBJECT)和动态物体(Category.VEHICLE, Category.PEOPLE)
        
        返回:
            filtered_points: np.array (M, 4), 更新后的带有语义的点云
        """
        
        if len(points) == 0:
            return points
        
        # 配置参数
        voxel_size = self.config['dynamic_filter_voxel_size']
        dynamic_ratio_threshold = self.config['dynamic_ratio_threshold']
        min_points_per_voxel = self.config['valid_voxel_point_threshold_dynamic_filter']
        
        # 动态物体的语义标签
        dynamic_labels = [Category.VEHICLE.value, Category.PEOPLE.value]
        
        # 1. 体素化
        xyz = points[:, :3]
        semantic = points[:, 3].astype(np.int32)
        
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
        
        # 构建体素字典：voxel_coord -> {'points_indices': [], 'semantics': [], 'visited': False, 'is_valid': False}
        voxel_dict = {}
        
        for i, voxel_coord in enumerate(voxel_coords):
            voxel_key = tuple(voxel_coord)
            
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points_indices': [],
                    'semantics': [],
                    'visited': False,
                    'is_valid': False,
                    'coord': voxel_coord
                }
            
            voxel_dict[voxel_key]['points_indices'].append(i)
            voxel_dict[voxel_key]['semantics'].append(semantic[i])
        
        # 1.5 标记有效体素（点数足够），并收集无效体素中的所有点
        invalid_voxel_points = []
        
        for voxel_key, voxel_data in voxel_dict.items():
            if len(voxel_data['points_indices']) >= min_points_per_voxel:
                voxel_data['is_valid'] = True
            else:
                # 无效体素：将其中所有点标记删除（不管是动态还是静态）
                point_indices = voxel_data['points_indices']
                invalid_voxel_points.extend(point_indices)
        
        # 2. 26邻域区域增长聚类（只对有效体素进行聚类）
        def get_26_neighbors(voxel_coord):
            """获取26邻域的体素坐标"""
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_coord = voxel_coord + np.array([dx, dy, dz])
                        neighbors.append(tuple(neighbor_coord))
            return neighbors
        
        clusters = []  # 存储聚类结果，每个聚类是体素key的列表
        
        # 遍历所有有效且未访问的体素进行区域增长
        for voxel_key, voxel_data in voxel_dict.items():
            if voxel_data['visited'] or not voxel_data['is_valid']:
                continue
            
            # 开始新的聚类
            cluster = []
            queue = deque([voxel_key])
            voxel_data['visited'] = True
            
            while queue:
                current_key = queue.popleft()
                cluster.append(current_key)
                
                current_data = voxel_dict[current_key]
                
                # 获取26邻域
                neighbors = get_26_neighbors(current_data['coord'])
                
                # 检查每个邻域体素
                for neighbor_key in neighbors:
                    if neighbor_key in voxel_dict:
                        neighbor_data = voxel_dict[neighbor_key]
                        
                        # 只有有效体素才能加入聚类
                        if not neighbor_data['visited'] and neighbor_data['is_valid']:
                            neighbor_data['visited'] = True
                            queue.append(neighbor_key)
            
            if cluster:
                clusters.append(cluster)
        
        # # ==================== 可视化聚类结果 ====================
        # print(f"聚类完成：共 {len(clusters)} 个聚类")
        
        # # 为每个聚类分配不同的颜色ID（用于可视化）
        # vis_points = []
        # for cluster_idx, cluster in enumerate(clusters):
        #     cluster_color = (cluster_idx % 20) + 1  # 使用1-20循环标记不同聚类
            
        #     for voxel_key in cluster:
        #         point_indices = voxel_dict[voxel_key]['points_indices']
        #         for idx in point_indices:
        #             x, y, z = points[idx, :3]
        #             vis_points.append([x, y, z, cluster_color])
        
        # if len(vis_points) > 0:
        #     vis_points_array = np.array(vis_points)
        #     vis_save_path = "/home/robot/data/Autolabel/Henan_dianzhan/test/clusters_vis.pcd"
        #     bin_to_pcd(vis_points_array, vis_save_path)
        #     print(f"✅ 聚类可视化保存至: {vis_save_path}")
        #     print(f"   - 总聚类数: {len(clusters)}")
        #     print(f"   - 总点数: {len(vis_points_array)}")
        # # =========================================================
        
        # 3. 统计每个聚类中动态物体体素的比例
        result_points = points.copy()
        
        # 获取最小有效体素数量阈值
        min_valid_voxel_count = self.config['valid_dynamic_voxel_size']
        
        # 记录需要删除的点的索引
        points_to_remove = []
        
        for cluster in clusters:
            total_voxel_count = len(cluster)
            
            # 3.1 如果聚类中的体素数量小于阈值，标记该聚类中的所有点为待删除
            if total_voxel_count < min_valid_voxel_count:
                for voxel_key in cluster:
                    point_indices = voxel_dict[voxel_key]['points_indices']
                    points_to_remove.extend(point_indices)
                continue  # 跳过后续处理
            
            # 3.2 统计该聚类中的动态物体体素数量和动态物体类别分布
            dynamic_voxel_count = 0
            
            # 统计聚类中各类动态物体的点数
            vehicle_count = 0
            people_count = 0
            
            for voxel_key in cluster:
                voxel_semantics = voxel_dict[voxel_key]['semantics']
                
                # 统计该体素中各类动态物体的点数
                for sem in voxel_semantics:
                    if sem == Category.VEHICLE.value:
                        vehicle_count += 1
                    elif sem == Category.PEOPLE.value:
                        people_count += 1
                
                # 如果该体素中有任何动态物体点，认为是动态体素
                if any(sem in dynamic_labels for sem in voxel_semantics):
                    dynamic_voxel_count += 1
            
            # 4. 如果动态物体体素比例 > 阈值，将整个聚类标记为动态物体
            dynamic_ratio = dynamic_voxel_count / total_voxel_count if total_voxel_count > 0 else 0
            
            if dynamic_ratio > dynamic_ratio_threshold:
                # 确定使用哪个动态物体类别（点数多的）
                if vehicle_count > people_count:
                    target_semantic = Category.VEHICLE.value
                elif people_count > vehicle_count:
                    target_semantic = Category.PEOPLE.value
                else:
                    # 如果相等，默认使用vehicle
                    target_semantic = Category.VEHICLE.value
                
                # 将该聚类中所有静态点标记为目标动态物体类别
                for voxel_key in cluster:
                    point_indices = voxel_dict[voxel_key]['points_indices']
                    for idx in point_indices:
                        # if result_points[idx, 3] == Category.STATIC_OBJECT.value:
                        result_points[idx, 3] = target_semantic
        
        # 5. 删除无效聚类中的所有点和无效体素中的所有点
        points_to_remove.extend(invalid_voxel_points)
        
        if points_to_remove:
            # 创建掩码标记要保留的点
            keep_mask = np.ones(len(result_points), dtype=bool)
            keep_mask[points_to_remove] = False
            result_points = result_points[keep_mask]
        
        return result_points


    def filter_dynamic_objects_people_prior(self, points):
        """
        通过体素化和聚类算法，滤除动态物体点（基于存在性的语义赋值）
        
        参数:
            points: np.array (N, 4), 不含地面点的点云 [x, y, z, semantic]
                    只包含静态物体(Category.STATIC_OBJECT)和动态物体(Category.VEHICLE, Category.PEOPLE)
        
        返回:
            filtered_points: np.array (M, 4), 更新后的带有语义的点云
            
        语义赋值逻辑:
            - 如果聚类中存在任意行人语义体素，则整个聚类赋值为行人
            - 否则整个聚类赋值为汽车
        """
        
        if len(points) == 0:
            return points
        
        # 配置参数
        voxel_size = self.config['dynamic_filter_voxel_size']
        min_points_per_voxel = self.config['valid_voxel_point_threshold_dynamic_filter']
        
        # 动态物体的语义标签
        dynamic_labels = [Category.VEHICLE.value, Category.PEOPLE.value]
        
        # 1. 体素化
        xyz = points[:, :3]
        semantic = points[:, 3].astype(np.int32)
        
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)
        
        # 构建体素字典：voxel_coord -> {'points_indices': [], 'semantics': [], 'visited': False, 'is_valid': False}
        voxel_dict = {}
        
        for i, voxel_coord in enumerate(voxel_coords):
            voxel_key = tuple(voxel_coord)
            
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = {
                    'points_indices': [],
                    'semantics': [],
                    'visited': False,
                    'is_valid': False,
                    'coord': voxel_coord
                }
            
            voxel_dict[voxel_key]['points_indices'].append(i)
            voxel_dict[voxel_key]['semantics'].append(semantic[i])
        
        # 1.5 标记有效体素（点数足够）
        for voxel_key, voxel_data in voxel_dict.items():
            if len(voxel_data['points_indices']) >= min_points_per_voxel:
                voxel_data['is_valid'] = True
        
        # 2. 26邻域区域增长聚类（只对有效体素进行聚类）
        def get_26_neighbors(voxel_coord):
            """获取26邻域的体素坐标"""
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor_coord = voxel_coord + np.array([dx, dy, dz])
                        neighbors.append(tuple(neighbor_coord))
            return neighbors
        
        clusters = []  # 存储聚类结果，每个聚类是体素key的列表
        
        # 遍历所有有效且未访问的体素进行区域增长
        for voxel_key, voxel_data in voxel_dict.items():
            if voxel_data['visited'] or not voxel_data['is_valid']:
                continue
            
            # 开始新的聚类
            cluster = []
            queue = deque([voxel_key])
            voxel_data['visited'] = True
            
            while queue:
                current_key = queue.popleft()
                cluster.append(current_key)
                
                current_data = voxel_dict[current_key]
                
                # 获取26邻域
                neighbors = get_26_neighbors(current_data['coord'])
                
                # 检查每个邻域体素
                for neighbor_key in neighbors:
                    if neighbor_key in voxel_dict:
                        neighbor_data = voxel_dict[neighbor_key]
                        
                        # 只有有效体素才能加入聚类
                        if not neighbor_data['visited'] and neighbor_data['is_valid']:
                            neighbor_data['visited'] = True
                            queue.append(neighbor_key)
            
            if cluster:
                clusters.append(cluster)

        # # ==================== 可视化聚类结果 ====================
        # print(f"聚类完成：共 {len(clusters)} 个聚类")
        
        # # 为每个聚类分配不同的颜色ID（用于可视化）
        # vis_points = []
        # for cluster_idx, cluster in enumerate(clusters):
        #     cluster_color = (cluster_idx % 20) + 1  # 使用1-20循环标记不同聚类
            
        #     for voxel_key in cluster:
        #         point_indices = voxel_dict[voxel_key]['points_indices']
        #         for idx in point_indices:
        #             x, y, z = points[idx, :3]
        #             vis_points.append([x, y, z, cluster_color])
        
        # if len(vis_points) > 0:
        #     vis_points_array = np.array(vis_points)
        #     vis_save_path = "/home/robot/data/Autolabel/AUTOLABEL_yuanqv_0208/debug/clip0004/clusters_visualization.pcd"
        #     bin_to_pcd(vis_points_array, vis_save_path)
        #     print(f"✅ 聚类可视化保存至: {vis_save_path}")
        #     print(f"   - 总聚类数: {len(clusters)}")
        #     print(f"   - 总点数: {len(vis_points_array)}")
        # # =========================================================
        # input("----------------------")
        
        # 3. 基于存在性判断聚类的语义类别
        result_points = points.copy()
        
        for cluster in clusters:
            # 检查聚类中是否存在行人体素
            has_people = False
            has_dynamic = False
            
            for voxel_key in cluster:
                voxel_semantics = voxel_dict[voxel_key]['semantics']
                
                # 检查该体素中是否有行人点
                if Category.PEOPLE.value in voxel_semantics:
                    has_people = True
                    has_dynamic = True
                    break  # 一旦发现行人，就可以确定整个聚类的语义
                
                # 检查是否有任何动态物体点
                if any(sem in dynamic_labels for sem in voxel_semantics):
                    has_dynamic = True
            
            # 4. 根据存在性判断结果，赋值语义
            if has_dynamic:
                # 如果存在行人体素，整个聚类赋值为行人；否则赋值为汽车
                if has_people:
                    target_semantic = Category.PEOPLE.value
                else:
                    target_semantic = Category.VEHICLE.value
                
                # 将该聚类中所有静态点标记为目标动态物体类别
                for voxel_key in cluster:
                    point_indices = voxel_dict[voxel_key]['points_indices']
                    for idx in point_indices:
                        # if result_points[idx, 3] == Category.STATIC_OBJECT.value:
                        result_points[idx, 3] = target_semantic
        
        return result_points



    def post_process_on_ego(self, points):
        '''
        对转换到ego坐标系下后并且crop后的多帧点云进行后处理
        
        '''
        # ----------------------------------- 拟合地面平面，后通过高度差阈值将static点重定义为地面点 -----------------------------------
        points = self.refine_ground_by_plane_fitting(points)

        # ----------------------------------- 通过地面mask 对地面点中的静态点进行去噪 -----------------------------------
        groud_denoised_points = self.bev_mask_ground_denoise(points)

        # ----------------------------------- 动态物体点聚类去除 -----------------------------------
        non_ground_points = groud_denoised_points[groud_denoised_points[:, 3] != Category.ROAD.value]

        # messi
        # static_and_dynamic_points = self.filter_dynamic_objects(non_ground_points)
        # static_and_dynamic_points = self.filter_dynamic_objects(non_ground_points)
        static_and_dynamic_points = non_ground_points
        # save_path = "/home/robot/data/Autolabel/AUTOLABEL_yuanqv_0208/debug/clip0004/dynamic_static.pcd"
        # bin_to_pcd(static_and_dynamic_points, save_path)
        # input("~~~~~~~~~~~~~~~~~")

        dynamic_points = static_and_dynamic_points[static_and_dynamic_points[:, 3] == Category.VEHICLE.value]

        static_points = static_and_dynamic_points[static_and_dynamic_points[:, 3] == Category.STATIC_OBJECT.value]
        ground_points = groud_denoised_points[groud_denoised_points[:, 3] == Category.ROAD.value]
        # ----------------------------------- 通过对静态点进行聚类算法去噪 -----------------------------------
        denoised_static = self.voxel_clustering_denoise(static_points)

        # ----------------------------------- 通过对地面点进行标准差算法去噪 -----------------------------------
        denoised_ground = self.ground_denoise_by_std_deviation(ground_points)

        full_static_points = np.vstack([denoised_ground, denoised_static])

        final_points = full_static_points

        # final_points = np.vstack([denoised_ground,denoised_static,dynamic_points])

        return final_points