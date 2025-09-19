import torch
import numpy as np
from scipy.spatial import KDTree

def chamfer_distance(x, y):
    """
    计算两组点云之间的Chamfer距离，并返回最近邻点的索引
    
    参数:
    x: numpy数组，形状为 (N, D)，第一组点云
    y: numpy数组，形状为 (M, D)，第二组点云
    
    返回:
    d1: float，从x到y的Chamfer距离
    d2: float，从y到x的Chamfer距离
    idx1: numpy数组，形状为 (N,)，x中每个点在y中的最近邻索引
    idx2: numpy数组，形状为 (M,)，y中每个点在x中的最近邻索引
    """
    # 构建KD树用于快速最近邻搜索
    tree_x = KDTree(x)
    tree_y = KDTree(y)
    
    # 对于y中的每个点，找到x中的最近邻
    dist_x, idx_x = tree_x.query(y)
    d1 = np.mean(dist_x**2)
    
    # 对于x中的每个点，找到y中的最近邻
    dist_y, idx_y = tree_y.query(x)
    d2 = np.mean(dist_y**2)
    
    return d1, d2, idx_y, idx_x

# 使用PyTorch版本的实现（如果需要GPU加速）
def chamfer_distance_torch(x, y):
    """
    使用PyTorch计算两组点云之间的Chamfer距离，并返回最近邻点的索引
    
    参数:
    x: torch.Tensor，形状为 (N, D)，第一组点云
    y: torch.Tensor，形状为 (M, D)，第二组点云
    
    返回:
    d1: torch.Tensor，从x到y的Chamfer距离
    d2: torch.Tensor，从y到x的Chamfer距离
    idx1: torch.Tensor，形状为 (N,)，x中每个点在y中的最近邻索引
    idx2: torch.Tensor，形状为 (M,)，y中每个点在x中的最近邻索引
    """
    # 计算所有点对之间的平方距离
    x = x.unsqueeze(1)  # (N, 1, D)
    y = y.unsqueeze(0)  # (1, M, D)
    dist = torch.sum((x - y) ** 2, dim=-1)  # (N, M)
    
    # 找到每个x点在y中的最近邻
    min_dist_x, idx1 = torch.min(dist, dim=1)
    d1 = torch.mean(min_dist_x)
    
    # 找到每个y点在x中的最近邻
    min_dist_y, idx2 = torch.min(dist, dim=0)
    d2 = torch.mean(min_dist_y)
    
    return d1, d2, idx1, idx2

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    x = np.random.rand(100, 3)  # 100个3D点
    y = np.random.rand(80, 3)   # 80个3D点
    
    # 计算Chamfer距离和索引
    d1, d2, idx1, idx2 = chamfer_distance(x, y)
    
    print(f"Chamfer distance x->y: {d1:.6f}")
    print(f"Chamfer distance y->x: {d2:.6f}")
    print(f"Indices of nearest points in y for each point in x: {idx1}")
    print(f"Indices of nearest points in x for each point in y: {idx2}")
    
    # 使用PyTorch版本
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: ", device)
    
    x_torch = torch.from_numpy(x).to(device)
    y_torch = torch.from_numpy(y).to(device)
    
    d1_t, d2_t, idx1_t, idx2_t = chamfer_distance_torch(x_torch, y_torch)
    
    print(f"\nPyTorch version:")
    print(f"Chamfer distance x->y: {d1_t.item():.6f}")
    print(f"Chamfer distance y->x: {d2_t.item():.6f}")
    print(f"Indices of nearest points in y for each point in x: {idx1_t.cpu().numpy()}")
    print(f"Indices of nearest points in x for each point in y: {idx2_t.cpu().numpy()}")