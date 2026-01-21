import numpy as np
import open3d as o3d
import os
import argparse
import struct

'''
将bin点云文件转换为Cloudcompare可以可视化的pcd文件
方便在Cloudcompare中可视化
且将bin中intensity列映射为点云的类别
'''


label_map = {
    0: 'ground',
    1: 'stair',
    2: 'other ds',
    2: 'not ds'
}

index_map = {
    10: 0,
    20: 1,
    30: 2,
    255: 3
}

index_map_array = np.zeros(256, dtype=np.uint8)
index_map_array[list(index_map.keys())] = list(index_map.values())


def bin_to_pcd_xyz(bin_file_path, pcd_file_path, dimension=3):
    """
    将.bin点云文件转换为.pcd格式
    
    参数:
    bin_file_path: .bin文件路径
    pcd_file_path: 输出.pcd文件路径
    dimension: 点云维度 (通常为4, 包含x,y,z,intensity)
    """
    
    # 从二进制文件读取数据
    points = np.fromfile(bin_file_path, dtype=np.float32)
    
    # 重塑数组为N×dimension的形状
    points = points.reshape(-1, dimension)
        
    num_points = points.shape[0]
    print(f"已读取 {num_points} 个点")
    
    with open(pcd_file_path, 'wb') as f:
        # 写入PCD文件头（文本部分）
#         header = f"""# .PCD v0.7 - Point Cloud Data file format
# VERSION 0.7
# FIELDS x y z
# SIZE 4 4 4
# TYPE F F F
# COUNT 1 1 1
# WIDTH {num_points}
# HEIGHT 1
# VIEWPOINT 0 0 0 1 0 0 0
# POINTS {num_points}
# DATA binary
# """   
        header = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA binary\n".format(num_points, num_points)
        f.write(header.encode('utf-8'))
        
        # 写入二进制点云数据
        # points[:, 3] = index_map_array[points[:, 3].astype(np.uint8)]
        points[:, :3].astype(np.float32).tofile(f)
        print(points[:5, :])

def bin_to_pcd(bin_file_path, pcd_file_path, dimension=4):
    """
    将.bin点云文件转换为.pcd格式
    
    参数:
    bin_file_path: .bin文件路径
    pcd_file_path: 输出.pcd文件路径
    dimension: 点云维度 (通常为4, 包含x,y,z,intensity)
    """
    
    # 从二进制文件读取数据
    points = np.fromfile(bin_file_path, dtype=np.float32)
    
    # 重塑数组为N×dimension的形状
    points = points.reshape(-1, dimension)
    print(np.unique(points[:,3]))
    
    assert points.shape[1] >= 4, "点云数据必须包含至少4列: x, y, z, intensity"
    
    num_points = points.shape[0]
    print(f"已读取 {num_points} 个点")
    
    with open(pcd_file_path, 'wb') as f:
        # 写入PCD文件头（文本部分）
#         header = f"""# .PCD v0.7 - Point Cloud Data file format
# VERSION 0.7
# FIELDS x y z intensity
# SIZE 4 4 4 4
# TYPE F F F F
# COUNT 1 1 1 1
# WIDTH {num_points}
# HEIGHT 1
# VIEWPOINT 0 0 0 1 0 0 0
# POINTS {num_points}
# DATA binary
# """   
        header = "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA binary\n".format(num_points, num_points)
        f.write(header.encode('utf-8'))
        
        # 写入二进制点云数据
        # points[:, 3] = index_map_array[points[:, 3].astype(np.uint8)]
        points[:, :4].astype(np.float32).tofile(f)
        print(points[:5, :])


if __name__ == "__main__":
    # 方法1: 单个文件转换
    bin_path = "/home/robot/data/tracking_debug/clip0000/lidar"
    pcd_path = "/home/robot/data/tracking_debug/clip0000/lidar_pcd"

    lidar_files = os.listdir(bin_path)
    timestamps = [f.split('.')[0] for f in lidar_files if f.endswith('.bin')]
    timestamps.sort(key=lambda x: float(x))

    for ts in timestamps:
        input_path = os.path.join(bin_path, ts+".bin")
        output_path = os.path.join(pcd_path, ts+".pcd")
        bin_to_pcd(input_path, output_path, dimension=4)

    # bin_path = "/home/robot/data/debug/clip0000/lidar/1532402927647951.bin"
    # pcd_path = "/home/robot/data/debug/clip0000/lidar/1532402927647951.pcd"

    # bin_to_pcd(bin_path, pcd_path, dimension=4)
    # bin_to_pcd_xyz(bin_path, pcd_path, dimension=3)
    