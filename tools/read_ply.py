import numpy as np
import open3d as o3d

import struct
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class GaussianPoint:
    """高斯泼溅点数据结构"""
    position: np.ndarray          # x, y, z
    normal: np.ndarray            # nx, ny, nz
    f_dc: np.ndarray              # f_dc_0, f_dc_1, f_dc_2 (球谐函数DC项)
    opacity: float                # 不透明度
    scale: np.ndarray             # scale_0, scale_1, scale_2
    rotation: np.ndarray          # rot_0, rot_1, rot_2, rot_3 (四元数)


def read_ply_simple(filepath):
    """
    简单读取PLY文件，使用Open3D
    
    Args:
        filepath (str): PLY文件路径
    
    Returns:
        numpy.ndarray: 点云数据
    """
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)

def visualize_ply(filepath):
    """
    读取并可视化PLY文件
    
    Args:
        filepath (str): PLY文件路径
    """
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(filepath)
    
    # 如果点云为空，输出提示信息
    if len(pcd.points) == 0:
        print("PLY文件中没有点云数据")
        return
    
    print(f"点云数据形状: {np.asarray(pcd.points).shape}")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Point Cloud Visualization", width=800, height=600)
    
    # 添加几何体
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 设置点的大小
    
    # 如果点云有颜色，启用颜色显示
    if pcd.has_colors():
        print("点云包含颜色信息")
    else:
        print("点云无颜色信息，使用默认颜色")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def read_gaussian_ply_numpy(file_path: str) -> dict:
    """
    使用numpy读取所有属性（最快的方法）
    
    参数:
        file_path: PLY文件路径
        
    返回:
        包含所有属性的字典
    """
    
    with open(file_path, 'rb') as f:
        # 读取header
        num_vertices = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
            elif line == 'end_header':
                break
        
        # 读取所有二进制数据
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4'),
            ('opacity', '<f4'),
            ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
            ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4')
        ])
        
        data = np.fromfile(f, dtype=dtype, count=num_vertices)
        
        # 整理数据
        result = {
            'positions': np.column_stack([data['x'], data['y'], data['z']]),
            'normals': np.column_stack([data['nx'], data['ny'], data['nz']]),
            'f_dc': np.column_stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']]),
            'opacity': data['opacity'],
            'scales': np.column_stack([data['scale_0'], data['scale_1'], data['scale_2']]),
            'rotations': np.column_stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']])
        }
        
        return result


if __name__ == "__main__":
    # points = read_ply_simple("/home/robot/Downloads/truck.ply")
    # print(points.shape)
    # visualize_ply("/home/robot/Downloads/truck.ply")

    data = read_gaussian_ply_numpy("/home/robot/Downloads/truck.ply")
    print(f"位置数据形状: {data['positions'].shape}")
    print(f"法线数据形状: {data['normals'].shape}")
    print(f"颜色数据形状: {data['f_dc'].shape}")
    print(f"缩放数据形状: {data['scales'].shape}")
    print(f"旋转数据形状: {data['rotations'].shape}")
