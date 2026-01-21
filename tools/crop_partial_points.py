import numpy as np

def bin_to_pcd(points, pcd_file_path, dimension=4):
    """
    将.bin点云文件转换为.pcd格式
    
    参数:
    bin_file_path: .bin文件路径
    pcd_file_path: 输出.pcd文件路径
    dimension: 点云维度 (通常为4, 包含x,y,z,intensity)
    """
    
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

def bin_to_pcd_xyz(points, pcd_file_path, dimension=3):
    """
    将.bin点云文件转换为.pcd格式
    
    参数:
    bin_file_path: .bin文件路径
    pcd_file_path: 输出.pcd文件路径
    dimension: 点云维度 (通常为4, 包含x,y,z,intensity)
    """
    
    # 重塑数组为N×dimension的形状
    points = points.reshape(-1, dimension)
    
    assert points.shape[1] == 3, "点云数据必须包含3列: x, y, z"
    
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

def extract_and_save_pointcloud(npz_file_path, output_bin_path, x_range=(-50, 50), y_range=(-50, 50)):
    """
    从npz文件中读取点云数据，截取指定x,y范围内的点云，并保存为bin文件
    
    Args:
        npz_file_path: npz文件路径
        output_bin_path: 输出bin文件路径
        x_range: x轴范围，默认(-50, 50)
        y_range: y轴范围，默认(-50, 50)
    """
    # 读取npz文件
    point_cloud = np.load(npz_file_path)['occ_gt']
    
    # 假设点云数据存储在'points'或'pointcloud'键中，具体键名需根据实际npz文件结构调整
    # 如果npz文件中有多个数组，需要查看具体键名
    # point_cloud = None
    # for key in data.files:
    #     if 'point' in key.lower() or 'cloud' in key.lower() or key == 'arr_0':
    #         point_cloud = data[key]
    #         break
    
    
    print(f"原始点云数据形状: {point_cloud.shape}")
    
    # 如果点云是多列的，通常前3列是x,y,z坐标
    if point_cloud.shape[1] >= 3:
        # 筛选x,y在指定范围内的点
        x_mask = (point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1])
        y_mask = (point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1])
        semantic_mask = point_cloud[:, 3] == 3
        
        # 组合筛选条件
        mask = x_mask & y_mask & semantic_mask
        
        # 应用筛选
        filtered_pointcloud = point_cloud[mask]
        
        print(f"筛选后点云数据形状: {filtered_pointcloud.shape}")
        
        # 保存为bin文件
        filtered_pointcloud.astype(np.float32).tofile(output_bin_path)
        print(f"点云已保存到: {output_bin_path}")
        output_pcd_path = "/home/robot/data/clip0/clip0000/find_high_ground.pcd"
        bin_to_pcd(filtered_pointcloud, output_pcd_path)
    else:
        print(f"点云数据格式不正确，列数: {point_cloud.shape[1]}")
        print("期望至少3列(x,y,z)")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    npz_file_path = "/home/robot/data/clip0/clip0000/occ_gt/1532402937698000.npz"
    output_bin_path = "/home/robot/data/clip0/clip0000/find_high_ground.bin"
    
    # 可自定义x,y范围
    extract_and_save_pointcloud(npz_file_path, output_bin_path, x_range=(-10, 30), y_range=(-30, 30))