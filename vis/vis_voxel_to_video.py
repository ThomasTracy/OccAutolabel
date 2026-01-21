'''
将多帧OCC语义体素可视化
并以一定的视角保存为图片并生成视频
'''

import cv2
import numpy as np
import open3d as o3d
import math
import colorsys
import os
from tqdm import tqdm

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

def get_scene_bounds(geometries):
    """
    获取所有几何体的边界
    """
    all_points = []
    for geom in geometries:
        if isinstance(geom, o3d.geometry.VoxelGrid):
            # 对于体素网格，获取所有体素中心点
            voxel_centers = []
            for voxel in geom.get_voxels():
                center = geom.get_voxel_center_coordinate(voxel.grid_index)
                voxel_centers.append(center)
            if voxel_centers:
                all_points.extend(voxel_centers)
        elif hasattr(geom, 'points'):
            # 对于点云等有点数据的几何体
            points = np.asarray(geom.points)
            all_points.extend(points)
        elif hasattr(geom, 'vertices'):
            # 对于网格等有顶点数据的几何体
            vertices = np.asarray(geom.vertices)
            all_points.extend(vertices)
    
    if not all_points:
        return np.array([0, 0, 0]), np.array([1, 1, 1])
    
    all_points = np.array(all_points)
    min_bound = np.min(all_points, axis=0)
    max_bound = np.max(all_points, axis=0)
    
    return min_bound, max_bound
def setup_camera_45_degree_view(vis):
    # 设置相机位置
    camera_pos = np.array([10, 0, -10])
    
    # 计算lookat点（场景中心）
    lookat_point = np.array([-10, 0, 0])
    
    # 设置相机参数
    ctr = vis.get_view_control()
    
    # 计算从相机位置指向场景的方向向量
    front_vector = lookat_point - camera_pos
    front_vector = front_vector / np.linalg.norm(front_vector)
    
    # 设置相机参数
    ctr.set_lookat(lookat_point)
    ctr.set_up([0, 0, 1])  # Z轴向上
    ctr.set_front(front_vector)
    ctr.set_zoom(0.1)  # 可以根据需要调整缩放级别
    
    return ctr

def setup_camera_top_view(vis):
    # 设置相机位置
    camera_pos = np.array([0, 0, -10])
    
    # 计算lookat点（场景中心）
    lookat_point = np.array([0, 0, 0])
    
    # 设置相机参数
    ctr = vis.get_view_control()
    
    # 计算从相机位置指向场景的方向向量
    front_vector = lookat_point - camera_pos
    front_vector = front_vector / np.linalg.norm(front_vector)
    
    # 设置相机参数
    ctr.set_lookat(lookat_point)
    ctr.set_up([0, 1, 0])  # Z轴向上
    ctr.set_front(front_vector)
    ctr.set_zoom(0.6)  # 可以根据需要调整缩放级别
    
    return ctr

def render_frame(vis, geometries, output_path, width=1200, height=800):
    """
    渲染当前帧并保存为图像
    """
    try:
        # 更新几何体
        for geom in geometries:
            vis.update_geometry(geom)
        
        # 设置视角
        # setup_camera_45_degree_view(vis)
        setup_camera_top_view(vis)
        
        # 更新渲染器
        vis.poll_events()
        vis.update_renderer()
        
        # 捕获屏幕
        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)
        
        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return True
    except Exception as e:
        print(f"渲染帧时出错: {e}")
        return False
def create_video_from_frames(frame_folder, output_video_path, fps=30):
    """
    将帧图像合成为视频
    """
    # 获取所有帧图像文件
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    
    if not frame_files:
        print("没有找到帧图像文件")
        return
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入所有帧
    for frame_file in tqdm(frame_files, desc="生成视频"):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"视频已保存至: {output_video_path}")

def visualize_voxels_multiframe(voxel_data_list, voxel_name_list, colors, voxel_size, output_dir="./frames", video_output="./voxel_video.mp4", fps=5):
    """
    可视化多帧体素数据并生成视频
    
    参数:
    voxel_data_list: 体素数据列表，每个元素为一帧的体素数据
    colors: 颜色映射数组
    voxel_size: 体素大小
    output_dir: 临时帧图像输出目录
    video_output: 输出视频路径
    fps: 视频帧率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800, visible=False)  # 设置为不可见以提高性能
    
    frames_data = []
    
    # 为每一帧创建几何体
    for i, voxel_data in enumerate(tqdm(voxel_data_list, desc="处理帧")):
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
        points = voxel_coords
        
        # 为每个点分配颜色
        point_colors = np.zeros((len(points), 3))
        
        for j, label in enumerate(voxel_labels):
            if label < len(colors):
                # 使用颜色映射，转换为0-1范围
                point_colors[j] = colors[label][:3] / 255.0
            # Unknown 类别用灰色表示
            elif label == 100:
                point_colors[j] = np.array([200, 200, 200]) / 255.0
            else:
                # 如果标签超出颜色映射范围，使用默认颜色（黑色）
                point_colors[j] = [1.0, 1.0, 1.0]
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # 创建体素网格
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        bounding_lines = create_voxel_bound_lines(voxel_grid)
        
        # 清空之前的几何体
        if i > 0:
            vis.clear_geometries()
        
        # 添加当前帧的几何体
        vis.add_geometry(voxel_grid)
        vis.add_geometry(bounding_lines)
        
        # 更新渲染
        vis.poll_events()
        vis.update_renderer()
        
        # 渲染并保存当前帧
        frame_path = os.path.join(output_dir, "{}.png".format(voxel_name_list[i]))
        render_frame(vis, [voxel_grid, bounding_lines], frame_path)
        
        frames_data.append((voxel_grid, bounding_lines))
    
    # 关闭可视化窗口
    vis.destroy_window()
    
    # 生成视频
    create_video_from_frames(output_dir, video_output, fps)
    
    # 可选：清理临时帧图像
    # import shutil
    # shutil.rmtree(output_dir)

def visualize_voxels_single_frame(voxel_data, colors, voxel_size):
    """
    单帧可视化（保持原有功能）
    """
    if voxel_data.shape[1] == 4:
        voxel_coords = voxel_data[:, :3]
        voxel_labels = voxel_data[:, 3].astype(int)
    elif voxel_data.shape[1] == 3:
        voxel_coords = voxel_data
        voxel_labels = np.zeros(len(voxel_data), dtype=int)
    else:
        raise ValueError("体素数据格式不正确，应为 [N, 3] 或 [N, 4]")
    
    points = voxel_coords
    point_colors = np.zeros((len(points), 3))
    
    for i, label in enumerate(voxel_labels):
        if label < len(colors):
            point_colors[i] = colors[label][:3] / 255.0
        # Unknown 类别用灰色表示
        elif label == 100:
            point_colors[i] = np.array([200, 200, 200]) / 255.0
        else:
            # 如果标签超出颜色映射范围，使用默认颜色（黑色）
            point_colors[i] = [1.0, 1.0, 1.0]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    bounding_lines = create_voxel_bound_lines(voxel_grid)
    
    o3d.visualization.draw_geometries([voxel_grid, bounding_lines], 
                                     window_name="Voxel Visualization",
                                     width=1200, 
                                     height=800)

def vis_voxel_multiframe(voxel_files, output_root):
    """
    处理多个体素文件生成视频
    """
    voxel_size = 0.2
    class_num = 6
    # colors = generate_uniform_colors_rgb(class_num)
    # colors = np.array(colors).astype(np.uint8)
    colors = np.array(
        [
            [21, 174, 103], # 绿色 静态物体
            [219, 79, 3], # 红色 车辆
            [250, 190, 0],   # 黄色  人
            [34, 174, 230], # 蓝色 路
            [200, 200, 200], # 灰色 空闲
        ]
    ).astype(np.uint8)
    
    # 加载所有体素数据
    voxel_data_list = []
    voxel_name_list = []
    for voxel_file in voxel_files:
        voxel_data = np.load(voxel_file)
        file_name = os.path.basename(voxel_file).replace('.npz', '')
        voxel_data = voxel_data['occ_gt']

        voxel_data = voxel_data[voxel_data[:,3]!=99]

        print(f"加载 {voxel_file}: {voxel_data.shape}")
        voxel_data_list.append(voxel_data)
        voxel_name_list.append(file_name)
    
    # 生成视频
    video_path = os.path.join(output_root, 'vis_video.mp4')
    frame_save_path = os.path.join(output_root, 'vis_frames')
    visualize_voxels_multiframe(voxel_data_list, voxel_name_list, colors, voxel_size, output_dir=frame_save_path, video_output=video_path)

def vis_voxel(voxel_file):
    """
    单帧可视化（保持原有功能）
    """
    voxel_size = 0.2
    class_num = 6
    colors = generate_uniform_colors_rgb(class_num)
    colors = np.array(colors).astype(np.uint8)
    voxel_data = np.load(voxel_file)
    voxel_data = voxel_data['occ_gt']
    print(voxel_data.shape)
    visualize_voxels_single_frame(voxel_data, colors, voxel_size)

def main(input_dir, output_root):
    npz_file_root = input_dir
    print(npz_file_root)
    
    # 确保文件存在
    existing_files = [os.path.join(npz_file_root, f) for f in os.listdir(npz_file_root)]
    sorted_files = sorted(existing_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    if sorted_files:
        vis_voxel_multiframe(sorted_files, output_root)
    else:
        print("未找到体素文件，请检查路径")

if __name__ == "__main__":

    # input_dir = '/home/robot/data/debug/clip0000/occ_gt'
    # output_dir = '/home/robot/data/debug/clip0000/vis_result'
    # main(input_dir, output_dir)

    multi_clip_root = '/home/robot/data/clip0'
    clips = os.listdir(multi_clip_root)
    for clip in clips:
        input_dir = os.path.join(multi_clip_root, clip, 'occ_gt')
        output_dir = os.path.join(multi_clip_root, clip, 'vis_result')
        main(input_dir, output_dir)

    # 单帧可视化
    # vis_voxel('/home/jszr/Data/OCC_Autolabel/DATA/tank/annotations/gt/1763027642111157.npz')

    # output_root = '/home/jszr/Data/OCC_Autolabel/DATA/tank/annotations'
    # npz_file_root = os.path.join(output_root, 'gt')
    # frame_save_root = os.path.join(output_root, 'vis_frames')
    # print(npz_file_root)
    
    # # 确保文件存在
    # existing_files = [os.path.join(npz_file_root, f) for f in os.listdir(npz_file_root)]
    # sorted_files = sorted(existing_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # if sorted_files:
    #     vis_voxel_multiframe(sorted_files, output_root)
    # else:
    #     print("未找到体素文件，请检查路径")