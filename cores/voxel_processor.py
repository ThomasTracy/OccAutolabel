import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings
from collections import defaultdict

class VoxelObservationModel:
    """
    基于体素化占据的传感器观测模型：使用射线投射法标记体素状态
    
    输入：已体素化的稠密占据Voxel，维度为N x 4 [x,y,z,semantic]
    其中x,y,z为体素中心点坐标（已体素化坐标），semantic为语义标签
    
    方法：
    1. 测量点所在的体素标记为"占据"（occupied）
    2. 从原点到测量点连线所经过的体素标记为"空闲"（free）
    3. 测量点之后的体素状态是"未知"（unknown）
    """
    
    def __init__(self):
        """
        初始化传感器观测模型
        注意：输入已经是体素化坐标，不需要体素大小参数
        """
        self.OCCUPIED = 2   # 占据状态
        self.FREE = 1       # 空闲状态
        self.UNKNOWN = 0    # 未知状态
        self.OCCUPIED_VOXELS = {}  # 存储占据体素及其语义信息
        
    def parse_occupied_voxels(self, voxel_data: np.ndarray) -> Dict[Tuple[int, int, int], int]:
        """
        解析输入的体素化占据数据
        
        参数：
        voxel_data: 体素数据，形状为(N, 4)，每列为[x, y, z, semantic]
                    x,y,z已经是体素坐标（整数）
        
        返回：
        occupied_voxels: 字典，键为体素坐标(x,y,z)的元组，值为语义标签
        """
        occupied_voxels = {}
        
        for i in range(voxel_data.shape[0]):
            x, y, z, semantic = voxel_data[i]
            voxel_key = (int(x), int(y), int(z))
            occupied_voxels[voxel_key] = int(semantic)
            
        self.OCCUPIED_VOXELS = occupied_voxels
        return occupied_voxels
    
    def bresenham_3d(self, start: Tuple[int, int, int], 
                     end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        3D Bresenham算法,获取两点之间直线经过的所有体素
        
        参数：
        start: 起点体素坐标 (x1, y1, z1)
        end: 终点体素坐标 (x2, y2, z2)
        
        返回：
        voxels: 从起点到终点经过的所有体素坐标列表（不包含终点）
        """
        voxels = []
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        # 如果起点和终点相同，返回空列表
        if start == end:
            return voxels
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        
        xs = 1 if x2 >= x1 else -1
        ys = 1 if y2 >= y1 else -1
        zs = 1 if z2 >= z1 else -1
        
        # 主驱动轴是x轴
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
                voxels.append((x1, y1, z1))
        # 主驱动轴是y轴
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
                voxels.append((x1, y1, z1))
        # 主驱动轴是z轴
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
                voxels.append((x1, y1, z1))
        
        # 移除最后一个点（终点）
        if len(voxels) > 0 and voxels[-1] == end:
            voxels = voxels[:-1]
            
        return voxels
    
    def ray_cast_from_sensor(self, occupied_voxel: Tuple[int, int, int],
                           sensor_origin: Tuple[int, int, int] = (0, 0, 0),
                           max_range: Optional[int] = None) -> List[Tuple[Tuple[int, int, int], int, int]]:
        """
        从传感器原点向占据体素发射射线
        
        参数：
        occupied_voxel: 占据体素坐标
        sensor_origin: 传感器原点坐标（体素坐标），默认为(0,0,0)
        max_range: 最大射线长度（体素单位），None表示无限制
        
        返回：
        ray_voxels: 列表，每个元素为(体素坐标, 状态, 语义标签)
                    语义标签仅对占据体素有效，空闲体素为-1
        """
        ray_voxels = []
        
        # 计算距离，用于判断是否超出最大范围
        dx = occupied_voxel[0] - sensor_origin[0]
        dy = occupied_voxel[1] - sensor_origin[1]
        dz = occupied_voxel[2] - sensor_origin[2]
        distance_sq = dx*dx + dy*dy + dz*dz
        
        if max_range is not None and distance_sq > max_range * max_range:
            return ray_voxels
        
        # 获取射线路径上的所有体素
        path_voxels = self.bresenham_3d(sensor_origin, occupied_voxel)
        
        # 标记路径上的体素为空闲（除终点外）
        for voxel in path_voxels:
            ray_voxels.append((voxel, self.FREE, -1))
        
        # 标记终点为占据，并添加语义信息
        semantic = self.OCCUPIED_VOXELS.get(occupied_voxel, -1)
        ray_voxels.append((occupied_voxel, self.OCCUPIED, semantic))
        
        return ray_voxels
    
    def compute_voxel_occupancy_grid(self, voxel_data: np.ndarray,
                                   sensor_origin: Tuple[int, int, int] = (0, 0, 0),
                                   max_range: Optional[int] = None,
                                   bounding_box: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None) -> Dict[str, np.ndarray]:
        """
        计算完整的体素占据栅格
        
        参数：
        voxel_data: 体素占据数据，形状为(N, 4)，[x,y,z,semantic]
        sensor_origin: 传感器原点坐标（体素坐标）
        max_range: 最大探测范围（体素单位）
        bounding_box: 感兴趣区域的边界框[(min_x, min_y, min_z), (max_x, max_y, max_z)]
        
        返回：
        occupancy_grid: 字典，包含：
                       'coordinates': 体素坐标数组 (M, 3)
                       'states': 体素状态数组 (M,) 0:未知, 1:空闲, 2:占据
                       'semantics': 语义标签数组 (M,)，-1表示无语义
        """
        # 解析占据体素
        occupied_voxels = self.parse_occupied_voxels(voxel_data)
        
        # 初始化体素状态字典
        # 使用defaultdict自动处理未见的体素为UNKNOWN
        voxel_states = defaultdict(lambda: (self.UNKNOWN, -1))
        
        print(f"处理 {len(occupied_voxels)} 个占据体素...")
        
        # 对每个占据体素进行射线投射
        for i, (occupied_voxel, semantic) in enumerate(occupied_voxels.items()):
            if i % 1000 == 0 and i > 0:
                print(f"  已处理 {i}/{len(occupied_voxels)} 个体素...")
            
            # 执行射线投射
            ray_results = self.ray_cast_from_sensor(occupied_voxel, sensor_origin, max_range)
            
            # 更新体素状态
            for voxel_coord, state, voxel_semantic in ray_results:
                # 只有当新状态优先级更高时更新
                # OCCUPIED(2) > FREE(1) > UNKNOWN(0)
                current_state, _ = voxel_states[voxel_coord]
                if state > current_state:
                    voxel_states[voxel_coord] = (state, voxel_semantic)
        
        print("射线投射完成")
        
        # 应用边界框过滤（如果提供）
        if bounding_box is not None:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = bounding_box
            filtered_voxels = {}
            for (x, y, z), (state, semantic) in voxel_states.items():
                if (min_x <= x <= max_x and 
                    min_y <= y <= max_y and 
                    min_z <= z <= max_z):
                    filtered_voxels[(x, y, z)] = (state, semantic)
            voxel_states = filtered_voxels
        
        # 转换为数组格式
        num_voxels = len(voxel_states)
        coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
        states = np.zeros(num_voxels, dtype=np.int32)
        semantics = np.zeros(num_voxels, dtype=np.int32)
        
        for i, ((x, y, z), (state, semantic)) in enumerate(voxel_states.items()):
            coordinates[i] = [x, y, z]
            states[i] = state
            semantics[i] = semantic
        
        return {
            'coordinates': coordinates,
            'states': states,
            'semantics': semantics
        }
    
    def create_dense_occupancy_grid(self, voxel_data: np.ndarray,
                                  grid_shape: Tuple[int, int, int],
                                  grid_origin: Tuple[int, int, int] = (0, 0, 0),
                                  sensor_origin: Tuple[int, int, int] = (0, 0, 0),
                                  max_range: Optional[int] = None) -> np.ndarray:
        """
        创建稠密的体素占据栅格（3D数组）
        
        参数：
        voxel_data: 体素占据数据
        grid_shape: 栅格形状 (size_x, size_y, size_z)
        grid_origin: 栅格原点在世界坐标系中的体素坐标
        sensor_origin: 传感器原点坐标
        max_range: 最大探测范围
        
        返回：
        occupancy_grid: 3D数组，形状为grid_shape，值为：
                       0: 未知, 1: 空闲, 2: 占据
        semantic_grid: 3D数组，形状为grid_shape，值为语义标签，-1表示无语义
        """
        # 计算稀疏的占据栅格
        sparse_result = self.compute_voxel_occupancy_grid(
            voxel_data, sensor_origin, max_range
        )
        
        # 创建稠密栅格
        size_x, size_y, size_z = grid_shape
        ox, oy, oz = grid_origin
        
        occupancy_grid = np.zeros(grid_shape, dtype=np.int32)
        semantic_grid = np.full(grid_shape, -1, dtype=np.int32)
        
        # 填充栅格
        for i in range(len(sparse_result['coordinates'])):
            x, y, z = sparse_result['coordinates'][i]
            state = sparse_result['states'][i]
            semantic = sparse_result['semantics'][i]
            
            # 转换到栅格坐标系
            grid_x = x - ox
            grid_y = y - oy
            grid_z = z - oz
            
            # 检查是否在栅格范围内
            if (0 <= grid_x < size_x and 
                0 <= grid_y < size_y and 
                0 <= grid_z < size_z):
                occupancy_grid[grid_x, grid_y, grid_z] = state
                semantic_grid[grid_x, grid_y, grid_z] = semantic
        
        return occupancy_grid, semantic_grid
    
    def analyze_occupancy_statistics(self, voxel_data: np.ndarray,
                                   sensor_origin: Tuple[int, int, int] = (0, 0, 0)) -> Dict[str, int]:
        """
        分析占据统计信息
        
        参数：
        voxel_data: 体素占据数据
        sensor_origin: 传感器原点坐标
        
        返回：
        stats: 统计信息字典
        """
        # 计算占据栅格
        result = self.compute_voxel_occupancy_grid(voxel_data, sensor_origin)
        
        states = result['states']
        
        # 统计各类体素数量
        total_voxels = len(states)
        unknown_count = np.sum(states == self.UNKNOWN)
        free_count = np.sum(states == self.FREE)
        occupied_count = np.sum(states == self.OCCUPIED)
        
        # 统计语义类别
        semantics = result['semantics']
        unique_semantics, semantic_counts = np.unique(semantics[semantics != -1], return_counts=True)
        
        stats = {
            'total_voxels': total_voxels,
            'unknown_voxels': int(unknown_count),
            'free_voxels': int(free_count),
            'occupied_voxels': int(occupied_count),
            'unknown_ratio': float(unknown_count / total_voxels) if total_voxels > 0 else 0,
            'free_ratio': float(free_count / total_voxels) if total_voxels > 0 else 0,
            'occupied_ratio': float(occupied_count / total_voxels) if total_voxels > 0 else 0,
            'semantic_classes': len(unique_semantics),
            'semantic_distribution': dict(zip(unique_semantics.tolist(), semantic_counts.tolist()))
        }
        
        return stats
    
    def visualize_2d_slice(self, voxel_data: np.ndarray,
                         slice_axis: str = 'z',
                         slice_value: int = 0,
                         sensor_origin: Tuple[int, int, int] = (0, 0, 0),
                         max_range: Optional[int] = 50) -> np.ndarray:
        """
        可视化2D切片
        
        参数：
        voxel_data: 体素占据数据
        slice_axis: 切片轴 'x', 'y', 或 'z'
        slice_value: 切片位置
        sensor_origin: 传感器原点坐标
        max_range: 最大探测范围
        
        返回：
        slice_grid: 2D切片数组
        """
        # 计算占据栅格
        result = self.compute_voxel_occupancy_grid(voxel_data, sensor_origin, max_range)
        
        # 提取指定切片的体素
        coordinates = result['coordinates']
        states = result['states']
        semantics = result['semantics']
        
        if slice_axis == 'x':
            mask = coordinates[:, 0] == slice_value
            slice_coords = coordinates[mask, 1:]  # 取y,z坐标
            axis_labels = ('Y', 'Z')
        elif slice_axis == 'y':
            mask = coordinates[:, 1] == slice_value
            slice_coords = coordinates[mask, [0, 2]]  # 取x,z坐标
            axis_labels = ('X', 'Z')
        else:  # 'z'
            mask = coordinates[:, 2] == slice_value
            slice_coords = coordinates[mask, :2]  # 取x,y坐标
            axis_labels = ('X', 'Y')
        
        slice_states = states[mask]
        slice_semantics = semantics[mask]
        
        if len(slice_coords) == 0:
            print(f"警告：在{slice_axis}={slice_value}处没有体素数据")
            return None
        
        # 计算切片边界
        min_coords = slice_coords.min(axis=0)
        max_coords = slice_coords.max(axis=0)
        
        size_x = max_coords[0] - min_coords[0] + 1
        size_y = max_coords[1] - min_coords[1] + 1
        
        # 创建切片网格
        slice_grid = np.zeros((size_x, size_y), dtype=np.int32)
        
        # 填充网格
        for i in range(len(slice_coords)):
            x, y = slice_coords[i]
            grid_x = x - min_coords[0]
            grid_y = y - min_coords[1]
            slice_grid[grid_x, grid_y] = slice_states[i]
        
        # 打印可视化
        print(f"\n{slice_axis.upper()}={slice_value} 平面切片可视化:")
        print(f"{axis_labels[1]}↓ {axis_labels[0]}→")
        
        for y in range(size_y-1, -1, -1):
            row_str = f"{y + min_coords[1]:3d} "
            for x in range(size_x):
                state = slice_grid[x, y]
                if state == self.OCCUPIED:
                    row_str += 'O'
                elif state == self.FREE:
                    row_str += '.'
                elif state == self.UNKNOWN:
                    row_str += '?'
                else:
                    row_str += ' '
            print(row_str)
        
        print("\n图例: O=占据, .=空闲, ?=未知")
        
        return slice_grid


# 使用示例
if __name__ == "__main__":
    # 1. 创建传感器观测模型
    voxel_model = VoxelObservationModel()
    
    # 2. 创建模拟的体素化占据数据 (N x 4)
    # 格式: [x, y, z, semantic]，x,y,z已经是体素坐标（整数）
    np.random.seed(42)
    
    # 生成一些占据体素
    num_occupied_voxels = 200
    voxel_data = np.zeros((num_occupied_voxels, 4), dtype=np.int32)
    
    # 生成不同位置的占据体素
    # 区域1: 远距离物体
    voxel_data[:50, 0] = np.random.randint(15, 25, 50)  # x: 15-24
    voxel_data[:50, 1] = np.random.randint(-5, 5, 50)   # y: -5~4
    voxel_data[:50, 2] = np.random.randint(0, 3, 50)    # z: 0~2
    voxel_data[:50, 3] = 1  # 语义标签1
    
    # 区域2: 中距离物体
    voxel_data[50:100, 0] = np.random.randint(8, 12, 50)  # x: 8-11
    voxel_data[50:100, 1] = np.random.randint(-3, 3, 50)  # y: -3~2
    voxel_data[50:100, 2] = np.random.randint(0, 5, 50)   # z: 0~4
    voxel_data[50:100, 3] = 2  # 语义标签2
    
    # 区域3: 近距离物体
    voxel_data[100:150, 0] = np.random.randint(3, 7, 50)   # x: 3-6
    voxel_data[100:150, 1] = np.random.randint(-2, 2, 50)  # y: -2~1
    voxel_data[100:150, 2] = np.random.randint(0, 2, 50)   # z: 0~1
    voxel_data[100:150, 3] = 3  # 语义标签3
    
    # 区域4: 近地面物体
    voxel_data[150:, 0] = np.random.randint(1, 4, 50)     # x: 1-3
    voxel_data[150:, 1] = np.random.randint(-1, 1, 50)    # y: -1~0
    voxel_data[150:, 2] = np.random.randint(0, 1, 50)     # z: 0
    voxel_data[150:, 3] = 4  # 语义标签4
    
    print(f"体素占据数据形状: {voxel_data.shape}")
    print(f"前10个占据体素:")
    for i in range(min(10, len(voxel_data))):
        print(f"  体素{voxel_data[i, :3]}, 语义标签: {voxel_data[i, 3]}")
    
    # 3. 计算完整的占据栅格
    print("\n计算占据栅格...")
    sensor_origin = (0, 0, 0)  # 传感器在原点
    max_range = 30  # 最大探测范围30个体素
    
    occupancy_result = voxel_model.compute_voxel_occupancy_grid(
        voxel_data, sensor_origin, max_range
    )
    
    print(f"\n占据栅格统计:")
    print(f"  总体素数: {len(occupancy_result['coordinates'])}")
    print(f"  占据体素: {np.sum(occupancy_result['states'] == voxel_model.OCCUPIED)}")
    print(f"  空闲体素: {np.sum(occupancy_result['states'] == voxel_model.FREE)}")
    print(f"  未知体素: {np.sum(occupancy_result['states'] == voxel_model.UNKNOWN)}")
    
    # 4. 分析统计信息
    print("\n分析统计信息...")
    stats = voxel_model.analyze_occupancy_statistics(voxel_data, sensor_origin)
    
    print(f"\n统计结果:")
    print(f"  总体素数: {stats['total_voxels']}")
    print(f"  未知体素: {stats['unknown_voxels']} ({stats['unknown_ratio']*100:.1f}%)")
    print(f"  空闲体素: {stats['free_voxels']} ({stats['free_ratio']*100:.1f}%)")
    print(f"  占据体素: {stats['occupied_voxels']} ({stats['occupied_ratio']*100:.1f}%)")
    print(f"  语义类别数: {stats['semantic_classes']}")
    print(f"  语义分布: {stats['semantic_distribution']}")
    
    # 5. 可视化z=0平面
    print("\n" + "="*60)
    print("Z=0平面可视化")
    print("="*60)
    
    voxel_model.visualize_2d_slice(
        voxel_data, 
        slice_axis='z', 
        slice_value=0,
        sensor_origin=sensor_origin,
        max_range=25
    )
    
    # 6. 创建稠密占据栅格
    print("\n" + "="*60)
    print("创建稠密占据栅格")
    print("="*60)
    
    grid_shape = (30, 20, 10)  # 30x20x10的栅格
    grid_origin = (-5, -10, -2)  # 栅格原点
    
    dense_occupancy, dense_semantic = voxel_model.create_dense_occupancy_grid(
        voxel_data, grid_shape, grid_origin, sensor_origin, max_range
    )
    
    print(f"稠密占据栅格形状: {dense_occupancy.shape}")
    print(f"稠密语义栅格形状: {dense_semantic.shape}")
    
    # 7. 测试单个射线投射
    print("\n" + "="*60)
    print("测试单个射线投射")
    print("="*60)
    
    # 选择一个占据体素
    test_occupied_voxel = tuple(voxel_data[75, :3])  # 第75个占据体素
    test_semantic = voxel_data[75, 3]
    
    print(f"测试占据体素: {test_occupied_voxel}, 语义标签: {test_semantic}")
    
    ray_results = voxel_model.ray_cast_from_sensor(
        test_occupied_voxel, sensor_origin, max_range
    )
    
    print(f"射线路径上的体素数量: {len(ray_results)}")
    print("射线路径上的体素状态:")
    
    for i, (voxel, state, semantic) in enumerate(ray_results):
        if i < 5 or i >= len(ray_results) - 5 or state == voxel_model.OCCUPIED:
            state_str = "占据" if state == voxel_model.OCCUPIED else "空闲"
            sem_str = f", 语义: {semantic}" if semantic != -1 else ""
            print(f"  体素{voxel}: {state_str}{sem_str}")
    
    if len(ray_results) > 10:
        print("  ... (中间省略)...")