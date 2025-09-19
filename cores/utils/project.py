import numpy as np

def project_point_cloud_to_camera(points,
                                  image_size,
                                  lidar2camera_T,
                                  intrinsic_matrix, 
                                  dist_coeffs=None):
    """
    将3D点云投影到相机2D图像平面,考虑相机畸变和坐标变换
    
    参数:
    points: 点云数据 (N, 3)
    intrinsic_matrix: 相机内参矩阵 (3, 3)
    dist_coeffs: 畸变系数 [k1, k2, p1, p2, k3]
    lidar2camera_T: 雷达到相机的变换矩阵 (4, 4)
    image_size: 图像尺寸 (height, width)
    """
    # 将点云从雷达坐标系变换到相机坐标系
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    points_camera = (lidar2camera_T @ points_homogeneous.T).T[:, :3]
    
    # 过滤掉相机后方的点 (z <= 0)
    mask = points_camera[:, 2] > 0
    points_camera = points_camera[mask]
    
    if len(points_camera) == 0:
        return np.array([]), np.array([])
    
    # 归一化坐标
    x = points_camera[:, 0] / points_camera[:, 2]
    y = points_camera[:, 1] / points_camera[:, 2]
    
    if dist_coeffs is not None:
        # 应用径向和切向畸变校正
        r2 = x**2 + y**2
        radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2 + dist_coeffs[4] * r2**3
        x_distorted = x * radial_distortion + 2 * dist_coeffs[2] * x * y + dist_coeffs[3] * (r2 + 2 * x**2)
        y_distorted = y * radial_distortion + dist_coeffs[2] * (r2 + 2 * y**2) + 2 * dist_coeffs[3] * x * y
    else:
        x_distorted, y_distorted = x, y
    
    # 应用内参矩阵投影到图像平面
    u = intrinsic_matrix[0, 0] * x_distorted + intrinsic_matrix[0, 2]
    v = intrinsic_matrix[1, 1] * y_distorted + intrinsic_matrix[1, 2]
    
    # 转换为整数坐标
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    
    # 过滤超出图像边界的点
    valid_indices = (u >= 0) & (u < image_size[1]) & (v >= 0) & (v < image_size[0])
    u = u[valid_indices]
    v = v[valid_indices]
    filtered_points = points_camera[valid_indices]
    coords_2d = np.stack([u, v], axis=1)
    
    return filtered_points, coords_2d