import numpy as np
import open3d as o3d


def voxelize_point_cloud(points, voxel_size=0.1):
    """
    对点云进行体素化
    """
    if len(points) == 0:
        print("没有点可以体素化")
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 体素化
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    return voxel_grid