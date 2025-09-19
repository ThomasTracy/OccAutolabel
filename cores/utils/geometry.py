import numpy as np


def remove_points_in_segmented_area(points, coords_2d, segmentation_mask, target_class=1):
    """
    根据图像分割掩码移除特定类别区域内的点
    
    参数:
    points: 点云数据 (N, 3)
    coords_2d: 点在图像上的坐标
    segmentation_mask: 图像分割掩码，与图像同尺寸，每个像素值为类别标签
    target_class: 要移除的目标类别, 默认为1
    """
    if coords_2d.shape[0] == 0:
        return np.array([]), []
    
    # 确保分割掩码的尺寸正确
    assert segmentation_mask.shape == segmentation_mask.shape, \
        "shape of segmentation_mask({}) does not match coords_2d size({})".format(segmentation_mask.shape, coords_2d.shape)
    
    # 检查哪些点在目标类别区域内
    valid_indices = []
    for i, (x, y) in enumerate(coords_2d):
        if segmentation_mask[x,y] != target_class:
            valid_indices.append(i)
    
    # 保留不在目标类别区域内的点
    remaining_points = points[valid_indices]
    
    print(f"移除类别{target_class}区域内的点后剩余点数: {len(remaining_points)}")
    print(f"移除的点数: {len(points) - len(remaining_points)}")
    
    return remaining_points
