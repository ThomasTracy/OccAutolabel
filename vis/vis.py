'''
单帧可视化 OCC 语义体素可视化
'''

import cv2
import numpy as np
import open3d as o3d
import math
import colorsys

def generate_uniform_colors_golden(color_num):
    """
    使用黄金角度在RGB立方体中均匀分布颜色
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    
    for i in range(color_num):
        hue = (i * golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    return colors

def generate_uniform_colors_rgb(color_num):
    """
    在RGB立方体中生成尽可能均匀分布的颜色
    """
    colors = []
    
    # 如果颜色数量较少，使用确定性方法
    if color_num <= 64:
        # 在RGB空间中找到尽可能均匀的分布
        side_length = max(2, round(color_num ** (1/3)))
        points = []
        
        for r in np.linspace(0, 1, side_length):
            for g in np.linspace(0, 1, side_length):
                for b in np.linspace(0, 1, side_length):
                    points.append((r, g, b))
        
        # 选择前color_num个点
        selected_points = points[:color_num]
        colors = [tuple(int(c * 255) for c in rgb) for rgb in selected_points]
    else:
        # 对于大量颜色，使用黄金角度方法
        colors = generate_uniform_colors_golden(color_num)
    
    return colors

def create_voxel_bound_lines(voxel_grid):
    voxel_size = voxel_grid.voxel_size
    lines = o3d.geometry.LineSet()
    vertices = []  # To store all vertices of all voxels
    lines_indices = [] # To store the connectivity for the lines

    for voxel in voxel_grid.get_voxels():
        # Get the center coordinate of the voxel
        center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

        # Calculate the 8 vertices of the voxel
        x, y, z = center
        dx = dy = dz = voxel_size / 2.0
        voxel_vertices = np.array([
            [x - dx, y - dx, z - dx], [x + dx, y - dx, z - dx],
            [x + dx, y + dx, z - dx], [x - dx, y + dx, z - dx],
            [x - dx, y - dx, z + dx], [x + dx, y - dx, z + dx],
            [x + dx, y + dx, z + dx], [x - dx, y + dx, z + dx]
        ])

        # Add the vertices to the list and record their indices
        current_vertex_offset = len(vertices)
        vertices.extend(voxel_vertices)

        # Define the 12 edges of the voxel by connecting the vertices
        # Example: A cube has 12 edges
        voxel_lines_indices = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
        ]) + current_vertex_offset
        lines_indices.extend(voxel_lines_indices)
    lines.points = o3d.utility.Vector3dVector(np.array(vertices))
    lines.lines = o3d.utility.Vector2iVector(np.array(lines_indices))
    lines.paint_uniform_color([0, 0, 0]) # Color the lines
    return lines

def visualize_voxels(voxel_data, colors, voxel_size):
    """
    使用Open3D可视化体素数据
    
    参数:
    voxel_data: 体素数据，形状为 [N, 4] 或 [N, 3],其中前3列为坐标,最后一列为语义标签
    colors: 颜色映射数组
    voxel_size: 体素大小
    """
    # 确保输入数据格式正确
    if voxel_data.shape[1] == 4:
        # 有语义标签
        voxel_coords = voxel_data[:, :3]
        voxel_labels = voxel_data[:, 3].astype(int)
    elif voxel_data.shape[1] == 3:
        # 只有坐标，没有语义标签
        voxel_coords = voxel_data
        voxel_labels = np.zeros(len(voxel_data), dtype=int)
    else:
        raise ValueError("体素数据格式不正确，应为 [N, 3] 或 [N, 4]")
    
    # 创建点云（体素中心点）
    # points = voxel_coords * voxel_size + voxel_size / 2  # 转换为实际坐标
    points = voxel_coords
    
    # 为每个点分配颜色
    point_colors = np.zeros((len(points), 3))
    
    for i, label in enumerate(voxel_labels):
        if label < len(colors):
            # 使用颜色映射，转换为0-1范围
            point_colors[i] = colors[label][:3] / 255.0
        # Unknown 类别用灰色表示
        elif label == 100:
            point_colors[i] = np.array([200, 200, 200]) / 255.0
        else:
            # 如果标签超出颜色映射范围，使用默认颜色（黑色）
            point_colors[i] = [1.0, 1.0, 1.0]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 创建体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    bounding_lines = create_voxel_bound_lines(voxel_grid)
    # 可视化
    o3d.visualization.draw_geometries([voxel_grid,bounding_lines], 
                                     window_name="Voxel Visualization",
                                     width=1200, 
                                     height=800)

def vis_voxel(voxel_file):
    colors = np.array(
        [
            [21, 174, 103], # 绿色 静态物体
            [219, 79, 3], # 红色 车辆
            [250, 190, 0],   # 黄色  人
            [34, 174, 230], # 蓝色 路
            [200, 200, 200], # 灰色 空闲
        ]
    ).astype(np.uint8)
    voxel_size = 0.1
    class_num = 6
    # colors = generate_uniform_colors_rgb(class_num)
    colors = np.array(colors).astype(np.uint8)
    voxel_data = np.load(voxel_file)
    voxel_data = voxel_data['occ_gt']
    print("loaded voxel data: ",voxel_data.shape)
    voxel_data = voxel_data[voxel_data[:,3]!=99]
    print("valid voxel data: ",voxel_data.shape)
    visualize_voxels(voxel_data, colors, voxel_size)


if __name__ == "__main__":
    # vis_voxel('/data/Data/OCC_Autolabel/data_clip0/occ_gt/1532402946797517.npz')
    # vis_voxel('/home/robot/data/nuscenes_mini_clips/clip0000/occ_gt/1532402935697945.npz')
    vis_voxel('/home/robot/data/clip0/clip0000/occ_gt/1532402941297218.npz')
    # vis_voxel('/home/robot/data/nuscenes_mini_clips/debug_ray_cast/without_underground.npz')