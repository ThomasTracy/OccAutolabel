import numpy as np
import open3d as o3d
import cv2


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
    points = voxel_coords * voxel_size + voxel_size / 2  # 转换为实际坐标
    
    # 为每个点分配颜色
    point_colors = np.zeros((len(points), 3))
    
    for i, label in enumerate(voxel_labels):
        if label < len(colors):
            # 使用颜色映射，转换为0-1范围
            point_colors[i] = colors[label][:3] / 255.0
        else:
            # 如果标签超出颜色映射范围，使用默认颜色（红色）
            point_colors[i] = [1.0, 0.0, 0.0]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 创建体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    # 可视化
    o3d.visualization.draw_geometries([voxel_grid], 
                                     window_name="Voxel Visualization",
                                     width=1200, 
                                     height=800)
    

def visualize_pc_on_image(image_data, projected_points):
    """
    可视化点云投影结果
    
    Args:
        first_frame_data: 第一帧数据字典
        projected_points: 投影点 (N, 3) [u, v, depth]
        valid_mask: 有效点掩码 (N,)
    """
    
    # 将PIL图像转换为OpenCV格式
    image = np.array(image_data)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 根据深度值归一化颜色
    depths = None
    norm_depths = None
    if projected_points.shape[1] == 3:
        depths = projected_points[:, 2]
        norm_depths = (depths - depths.min()) / (depths.max() - depths.min())
    
    # 在图像上绘制点
    for i in range(projected_points.shape[0]):
        u, v = projected_points[i][:2]
        u, v = int(u), int(v)
        
        # 确保点在图像范围内
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            # 根据深度值设置颜色（近处为红色，远处为蓝色）
            if depths is not None:
                color_value = int(norm_depths[i] * 255)
                b = color_value
                g = 0
                r = 255 - color_value
            else:
                b, g, r = 0, 255, 0
            
            # 绘制点
            cv2.circle(image, (u, v), 3, (b, g, r), -1)
    
    cv2.imshow('Point Cloud on Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
