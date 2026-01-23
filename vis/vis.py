"""
单帧可视化 OCC 语义体素可视化
优化版本：使用向量化操作提升性能
"""

import numpy as np
import open3d as o3d
from typing import Dict, Tuple
import time


def create_voxel_bound_lines_fast(voxel_coords: np.ndarray, voxel_size: float) -> o3d.geometry.LineSet:
    """
    快速创建体素边界线（向量化版本）
    
    Args:
        voxel_coords: 体素中心坐标，形状为 [N, 3]
        voxel_size: 体素大小
    
    Returns:
        LineSet对象，包含所有体素的边界线
    """
    n_voxels = len(voxel_coords)
    half_size = voxel_size / 2.0
    
    # 定义立方体的8个顶点相对于中心的偏移量
    vertex_offsets = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 底面
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # 顶面
    ]) * half_size
    
    # 向量化计算所有体素的所有顶点：(N, 8, 3)
    all_vertices = voxel_coords[:, np.newaxis, :] + vertex_offsets[np.newaxis, :, :]
    all_vertices = all_vertices.reshape(-1, 3)  # (N*8, 3)
    
    # 定义立方体的12条边（顶点索引）
    edge_indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 竖边
    ])
    
    # 为每个体素生成边的索引：(N, 12, 2)
    base_indices = np.arange(n_voxels)[:, np.newaxis, np.newaxis] * 8
    all_lines = base_indices + edge_indices[np.newaxis, :, :]
    all_lines = all_lines.reshape(-1, 2)  # (N*12, 2)
    
    # 创建LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_vertices)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.paint_uniform_color([0, 0, 0])
    
    return line_set


def assign_colors_vectorized(voxel_labels: np.ndarray, 
                            color_map: Dict[int, Tuple[int, int, int]],
                            default_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    向量化方式为体素分配颜色
    
    Args:
        voxel_labels: 语义标签数组，形状为 [N]
        color_map: 语义ID到颜色的映射字典 {semantic_id: (R, G, B)}
        default_color: 默认颜色（当标签不在映射中时使用）
    
    Returns:
        颜色数组，形状为 [N, 3]，范围 [0, 1]
    """
    n_points = len(voxel_labels)
    
    # 获取所有唯一的标签
    unique_labels = np.unique(voxel_labels)
    max_label = max(max(color_map.keys()), voxel_labels.max())
    
    # 创建完整的颜色查找表（包括所有可能的标签）
    color_lut = np.ones((max_label + 1, 3), dtype=np.float32)
    color_lut *= np.array(default_color) / 255.0
    
    # 填充已知标签的颜色
    for label, color in color_map.items():
        if label <= max_label:
            color_lut[label] = np.array(color) / 255.0
    
    # 使用高级索引一次性分配所有颜色
    point_colors = color_lut[voxel_labels]
    
    return point_colors


def visualize_voxels(voxel_data: np.ndarray, 
                    color_map: Dict[int, Tuple[int, int, int]], 
                    voxel_size: float,
                    window_name: str = "Voxel Visualization",
                    window_size: Tuple[int, int] = (1200, 800)) -> None:
    """
    高效可视化体素数据（优化版本）
    
    Args:
        voxel_data: 体素数据，形状为 [N, 4] (x, y, z, semantic) 或 [N, 3] (x, y, z)
        color_map: 语义ID到RGB颜色的映射字典
        voxel_size: 体素大小
        window_name: 窗口标题
        window_size: 窗口大小 (width, height)
    """
    start_time = time.time()
    
    # 解析数据
    if voxel_data.shape[1] == 4:
        voxel_coords = voxel_data[:, :3]
        voxel_labels = voxel_data[:, 3].astype(np.int32)
    elif voxel_data.shape[1] == 3:
        voxel_coords = voxel_data
        voxel_labels = np.zeros(len(voxel_data), dtype=np.int32)
    else:
        raise ValueError(f"体素数据格式不正确，应为 [N, 3] 或 [N, 4]，当前为 {voxel_data.shape}")
    
    print(f"[Timer] 数据解析: {time.time() - start_time:.3f}s")
    
    # 向量化分配颜色
    t1 = time.time()
    point_colors = assign_colors_vectorized(voxel_labels, color_map)
    print(f"[Timer] 颜色分配: {time.time() - t1:.3f}s")
    
    # 创建点云
    t2 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_coords.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))
    print(f"[Timer] 创建点云: {time.time() - t2:.3f}s")
    
    # 创建体素网格
    t3 = time.time()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    print(f"[Timer] 创建体素网格: {time.time() - t3:.3f}s")
    
    # 从VoxelGrid中提取实际的体素中心坐标（确保与渲染的体素对齐）
    t4 = time.time()
    actual_voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) 
                                     for voxel in voxel_grid.get_voxels()])
    
    # 调试信息：检查输入坐标与VoxelGrid生成的坐标差异
    if len(actual_voxel_centers) > 0 and len(voxel_coords) > 0:
        # 检查前几个点的差异
        n_sample = min(5, len(voxel_coords), len(actual_voxel_centers))
        print(f"\n[Debug] 坐标对比（前{n_sample}个）:")
        print("  输入坐标 -> VoxelGrid中心坐标")
        for i in range(n_sample):
            if i < len(voxel_coords) and i < len(actual_voxel_centers):
                print(f"  {voxel_coords[i]} -> {actual_voxel_centers[i]}")
        
        # 计算整体差异
        if len(voxel_coords) == len(actual_voxel_centers):
            diff = np.abs(voxel_coords - actual_voxel_centers).max()
            print(f"[Debug] 最大坐标差异: {diff:.6f}")
            if diff < 1e-6:
                print("[Debug] ✓ 输入坐标与VoxelGrid坐标完全匹配（数据已经是体素中心）")
            else:
                print(f"[Debug] ⚠ 输入坐标与VoxelGrid坐标存在差异（可能需要量化）")
    
    print(f"[Debug] VoxelGrid包含 {len(actual_voxel_centers)} 个体素")
    
    # 使用实际的体素中心坐标创建边界线
    bounding_lines = create_voxel_bound_lines_fast(actual_voxel_centers, voxel_size)
    print(f"[Timer] 创建边界线: {time.time() - t4:.3f}s")
    
    print(f"[Timer] 总耗时: {time.time() - start_time:.3f}s")
    print(f"[Info] 输入数据点数: {len(voxel_coords)}, 实际体素数: {len(actual_voxel_centers)}")
    
    # 可视化
    o3d.visualization.draw_geometries(
        [voxel_grid, bounding_lines], 
        window_name=window_name,
        width=window_size[0], 
        height=window_size[1]
    )


def load_and_visualize(voxel_file: str, 
                      color_map: Dict[int, Tuple[int, int, int]],
                      voxel_size: float = 0.1,
                      filter_label: int = 99) -> None:
    """
    加载并可视化体素数据
    
    Args:
        voxel_file: npz文件路径
        color_map: 语义到颜色的映射
        voxel_size: 体素大小
        filter_label: 需要过滤的标签（不显示）
    """
    print(f"加载文件: {voxel_file}")
    voxel_data = np.load(voxel_file)['occ_gt']
    print(f"原始数据形状: {voxel_data.shape}")
    
    # 过滤特定标签
    if filter_label is not None:
        mask = voxel_data[:, 3] != filter_label
        voxel_data = voxel_data[mask]
        print(f"过滤后数据形状: {voxel_data.shape}")
    
    visualize_voxels(voxel_data, color_map, voxel_size)


if __name__ == "__main__":
    import argparse
    
    # 在主函数中定义语义到颜色的映射
    # 格式: {semantic_id: (R, G, B)}
    SEMANTIC_COLOR_MAP = {
        0: (21, 174, 103),    # 绿色 - 静态物体
        1: (219, 79, 3),      # 红色 - 车辆
        2: (250, 190, 0),     # 黄色 - 人
        3: (34, 174, 230),    # 蓝色 - 路
        99: (200, 200, 200),   # 灰色 - 空闲
        100: (150, 150, 150), # 浅灰色 - Unknown
    }
    
    parser = argparse.ArgumentParser(description='体素数据可视化工具（优化版本）')
    parser.add_argument('--voxel_file', type=str, 
                       default='/home/robot/data/clip0/clip0000/occ_gt/1532402941297218.npz',
                       help='体素数据文件路径 (.npz格式)')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                       help='体素大小')
    parser.add_argument('--filter_label', type=int, default=99,
                       help='需要过滤的标签（不显示），设置为None则不过滤')
    args = parser.parse_args()
    
    print("=" * 60)
    print("语义颜色映射配置:")
    for label, color in sorted(SEMANTIC_COLOR_MAP.items()):
        print(f"  标签 {label:3d}: RGB{color}")
    print("=" * 60)
    
    # 调用可视化函数
    load_and_visualize(
        voxel_file=args.voxel_file,
        color_map=SEMANTIC_COLOR_MAP,
        voxel_size=args.voxel_size,
        filter_label=args.filter_label
    )


# 基本使用
# python vis/vis.py --voxel_file path/to/data.npz

# 自定义参数
# python vis/vis.py --voxel_file data.npz --voxel_size 0.2 --filter_label 99