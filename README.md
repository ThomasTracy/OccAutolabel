# OCC Autolabel - 3D语义占据网格自动标注系统

基于多模态融合的3D场景语义占据（Occupancy）自动标注工具，通过LiDAR点云、多相机图像和语义分割掩码，自动生成高质量的3D语义占据网格标注数据。

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [数据格式](#数据格式)
- [配置说明](#配置说明)
- [工具集](#工具集)
- [可视化](#可视化)

---

## 项目概述

本项目实现了基于SAM3（Segment Anything Model 3）语义分割掩码的3D占据网格自动标注流程。系统通过融合LiDAR点云和多相机语义掩码，生成稠密的3D语义占据体素网格，支持自动驾驶、机器人等领域的3D场景理解任务。

### 主要应用场景
- 自动驾驶感知数据集标注
- 机器人导航场景理解
- 3D场景重建与语义分割
- 占据网格预测模型训练数据生成

---

## 核心特性

### 🎯 多模态数据融合
- **LiDAR点云处理**：支持多帧点云拼接与坐标系转换
- **多相机投影**：支持6个相机视角的点云投影与语义标注
- **语义掩码融合**：基于SAM3等分割模型的掩码进行语义标注

### 🔧 智能处理流程
- **动态物体去噪**：基于体素密度和空间聚类的动态物体噪声过滤
- **地面点处理**：自动识别和标注地面点及盲区处理
- **射线投射**：基于传感器观测模型判断Free/Unknown空间
- **多帧融合**：时序点云配准与融合，提升数据完整性

### 📊 语义类别支持
- **静态物体** (Static Object)：建筑物、障碍物等
- **车辆** (Vehicle)：各类机动车
- **行人** (People)：人类
- **道路** (Road)：可行驶路面
- **空闲空间** (Free)：可通行的空白区域
- **未知区域** (Unknown)：传感器未观测到的区域

---

## 系统架构

```
occ_autolabel/
├── cores/                          # 核心功能模块
│   ├── autolabel/                  # 自动标注实现
│   │   ├── occ_autolabel_base.py          # 基类：基础功能
│   │   ├── occ_autolabel_SAM3_mask.py     # SAM3掩码标注（主流程）
│   │   ├── occ_autolabel_semantic_points.py  # 语义点云标注
│   │   └── occ_autolabel_with_ue_pose.py  # UE模拟器位姿版本
│   ├── config/                     # 配置管理
│   │   ├── config.py              # 配置类
│   │   └── dataclass.py           # 数据类定义
│   ├── dataset/                    # 数据集解析
│   │   ├── dataset.py             # 通用数据集
│   │   └── parse_nuscenes_data.py # NuScenes格式解析
│   ├── utils/                      # 工具函数
│   │   ├── projector.py           # 坐标投影工具
│   │   ├── geometry.py            # 几何处理
│   │   ├── voxelize.py            # 体素化
│   │   └── visualize.py           # 可视化工具
│   ├── Processor.py                # 预处理与后处理器
│   ├── voxel_processor.py          # 体素观测模型（射线投射）
│   └── category.py                 # 语义类别定义
├── configs/                        # 配置文件
│   ├── config_sam3_nuscenes.yaml  # SAM3+NuScenes配置
│   ├── config.yaml                # 通用配置
│   └── label_mapping_nuscenes.yaml # 语义标签映射
├── tools/                          # 辅助工具
│   ├── bin2pcd.py                 # 点云格式转换
│   ├── lidar_on_cam_video.py      # LiDAR-Camera融合视频生成
│   ├── calibration_generator.py   # 标定文件生成
│   └── coco_anno_generator.py     # COCO格式标注生成
├── extensions/                     # C++扩展
│   └── chamfer_dist/              # Chamfer距离计算（CUDA加速）
├── vis_voxel_to_video.py          # 体素可视化视频生成
├── vis.py                          # 交互式可视化
└── run_SAM3.py                    # 主启动脚本
```

---

## 环境要求

### 系统要求
- **操作系统**：Linux (Ubuntu 18.04+推荐)
- **Python**：3.8 - 3.11
- **CUDA**：10.2+ (用于Chamfer距离加速，可选)

### 依赖库
```bash
# 核心依赖
numpy>=1.19.0
opencv-python>=4.5.0
open3d>=0.13.0
pyyaml>=5.4
scipy>=1.7.0
pandas>=1.3.0
tqdm>=4.60.0

# 深度学习（可选）
torch>=1.9.0  # 用于Chamfer距离计算
```

---

## 安装

### 1. 克隆项目
```bash
git clone http://gitlab.jszr.com/songyuduo/occ_autolabel.git
cd occ_autolabel
```

### 2. 创建虚拟环境（推荐）
```bash
conda create -n occ_autolabel python=3.9
conda activate occ_autolabel
```

### 3. 安装依赖
```bash
pip install numpy opencv-python open3d pyyaml scipy pandas tqdm
pip install torch  # 如需CUDA加速
```

### 4. 编译Chamfer距离扩展（可选，提升性能）
```bash
cd extensions/chamfer_dist
python setup.py install
cd ../..
```

---

## 快速开始

### 1. 准备数据

按照以下目录结构组织数据：
```
data_root/
├── clip0000/
│   ├── lidar/                    # LiDAR点云 (.bin文件)
│   │   ├── 1532402935697945.bin
│   │   └── ...
│   ├── camera/                   # 多相机图像
│   │   ├── CAM_FRONT/
│   │   │   ├── 1532402935697945.jpg
│   │   │   └── ...
│   │   ├── CAM_BACK/
│   │   └── ...
│   ├── mask/                     # 语义分割掩码 (SAM3生成)
│   │   ├── CAM_FRONT/
│   │   │   ├── mask_1.npy      # 类别1的掩码
│   │   │   ├── mask_2.npy      # 类别2的掩码
│   │   │   └── ...
│   │   └── ...
│   ├── calibration/             # 标定文件
│   │   └── calibration.json
│   └── pose/                    # 位姿数据
│       └── pose.json
└── clip0001/
    └── ...
```

### 2. 配置参数

编辑 `configs/config_sam3_nuscenes.yaml`：
```yaml
# 数据路径
data_root: "/path/to/your/data_root"
save_path: "occ_gt"  # 相对于每个clip的保存路径

# 标签映射
label_map: "configs/label_mapping_nuscenes.yaml"

# 相机类型
camera_types: ["CAM_BACK","CAM_BACK_RIGHT","CAM_FRONT_LEFT",
               "CAM_BACK_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]

# 体素化参数
voxel_size: 0.1              # 体素尺寸(米)
voxel_range: [-10, -15, -1, 20, 15, 2]  # [x_min, y_min, z_min, x_max, y_max, z_max]

# 多帧融合参数
point_multi_frame_num: 10    # 融合帧数
multi_frame_radius: 100      # 融合半径(帧数),100表示当前帧前后100帧，共201帧进行融合

# 动态物体去噪
dynamic_denoise_voxel_size: 0.3
dynamic_denoise_dilate_radius: 2
```

### 3. 运行标注

```bash
# 修改run_SAM3.py中的配置路径
python run_SAM3.py
```

### 4. 输出结果

生成的占据网格保存为 `.npz` 文件：
```python
# 读取结果
data = np.load('data_root/clip0000/occ_gt/1532402935697945.npz')
occ_gt = data['occ_gt']  # shape: (N, 4) [x, y, z, semantic_label]
```

---

## 数据格式

### LiDAR点云格式 (.bin)
```python
# 二进制文件，每个点4或5个float32值
# 格式：[x, y, z, intensity] 或 [x, y, z, intensity, semantic]
points = np.fromfile('lidar.bin', dtype=np.float32).reshape(-1, 4)
```

### 标定文件格式 (calibration.json)
```json
{
  "lidar": {
    "lidar2ego": {
      "rotation": [w, x, y, z],      // 四元数
      "translation": [x, y, z]        // 平移向量
    }
  },
  "CAM_FRONT": {
    "cam2ego": {
      "rotation": [w, x, y, z],
      "translation": [x, y, z]
    },
    "camera_intrinsic": [
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1]
    ]
  }
}
```

### 位姿文件格式 (pose.json)
```json
{
  "1532402935697945": {
    "translation": [x, y, z],        // 世界坐标
    "rotation": [w, x, y, z]         // 四元数
  }
}
```

### 语义掩码格式 (.npy)
```python
# NumPy数组，形状为 (H, W)
# 每个像素值对应语义类别ID
mask = np.load('mask/CAM_FRONT/mask_1.npy')
```

---

## 配置说明

### 关键参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `voxel_size` | float | 0.1 | 体素边长(米)，越小越精细但计算量大 |
| `voxel_range` | list | [-10,-15,-1,20,15,2] | 感兴趣区域范围(米) |
| `point_multi_frame_num` | int | 10 | 多帧点云融合帧数 |
| `sensor_origin` | list | [0,0,1] | 射线投射起点(ego坐标系) |
| `valid_voxel_point_threshold` | int | 10 | 有效体素最小点数阈值 |
| `dynamic_denoise_voxel_size` | float | 0.3 | 动态去噪体素尺寸 |
| `ground_height_threshold` | float | 0.2 | 地面高度阈值(米) |
| `ego_range` | list | [-1,-1,3,1] | 自车区域范围 |

---

## 工具集

### 1. 点云格式转换
```bash
python tools/bin2pcd.py <input.bin> <output.pcd>
```

### 2. LiDAR-Camera投影视频生成
```bash
python tools/lidar_on_cam_video.py --root <data_path>
```

### 3. 标定文件生成
```bash
python tools/calibration_generator.py
```

### 4. NuScenes数据解析
```bash
python cores/dataset/parse_nuscenes_data.py
```

---

## 可视化

### 1. 交互式可视化（单帧）
```bash
python vis.py
# 修改文件中的voxel_file路径
```
**交互操作**：
- 鼠标拖拽：旋转视角
- 滚轮：缩放
- `Ctrl+S`：保存相机视角配置

### 2. 生成可视化视频（多帧）
```bash
python vis_voxel_to_video.py
# 配置input_dir和output_dir
```
生成包含所有帧的MP4视频。

### 3. LiDAR投影到相机视频
```bash
python tools/lidar_on_cam_video.py
```

---

## 核心算法流程

### 主流程（OCCAutolabelwithMask）

1. **多帧点云拼接**
   - 加载连续N帧LiDAR点云
   - 通过位姿变换统一到参考帧坐标系
   - 去除自车点云和超出范围的点

2. **语义标注**
   - 将点云投影到各相机图像平面
   - 根据SAM3分割掩码为点赋予语义标签
   - 多相机融合，取最大语义概率

3. **坐标变换**
   - 点云从lidar坐标系→ego坐标系→世界坐标系
   - 转换到当前参考帧ego坐标系
   - 裁剪到感兴趣区域(voxel_range)

4. **后处理**
   - 地面盲区语义修正
   - 动态物体去噪（基于体素密度和膨胀）
   - 地面点过滤

5. **体素化与占据判断**
   - 稠密体素化：最近邻插值分配语义
   - 射线投射：标记Free/Unknown空间
   - 生成最终占据网格(N×4: x,y,z,semantic)

6. **保存结果**
   - 保存为.npz格式
   - 可选：保存为.ply用于可视化

### 动态物体去噪算法
```
1. 粗体素化分组
2. 计算每个体素中心的密度
3. 形态学膨胀操作
4. 过滤低密度孤立体素
5. 聚类提取主要物体
```

### 射线投射算法（Free/Unknown判断）
```
基于VoxelObservationModel：
- Occupied: 点云测量点所在体素
- Free: 传感器原点到测量点连线上的体素
- Unknown: 测量点之外未被观测的体素
```

---

## 性能优化建议

1. **使用Chamfer距离CUDA加速**：编译安装extensions/chamfer_dist
2. **调整体素尺寸**：增大voxel_size可显著减少计算量
3. **减少融合帧数**：降低point_multi_frame_num
4. **并行处理**：多clip可并行处理

---

## 常见问题

### Q1: 内存溢出怎么办？
A: 减小`voxel_range`范围或增大`voxel_size`，减少`point_multi_frame_num`。

### Q2: 为什么生成的占据网格不完整？
A: 检查相机标定是否正确，确保mask文件存在且格式正确。

### Q3: 如何支持自定义数据集？
A: 参考`cores/dataset/parse_nuscenes_data.py`，实现数据解析器，转换为标准格式。

### Q4: 如何调整语义类别？
A: 修改`configs/label_mapping_nuscenes.yaml`和`cores/category.py`。

---

## 引用

如果本项目对您的研究有帮助，请引用：
```bibtex
@software{occ_autolabel,
  title={OCC Autolabel: 3D Semantic Occupancy Auto-Labeling System},
  author={Your Team},
  year={2026},
  url={http://gitlab.jszr.com/songyuduo/occ_autolabel}
}
```

---

## 许可证

本项目采用内部许可，未经授权不得用于商业用途。

## 联系方式

- **项目维护**：songyuduo
- **问题反馈**：通过GitLab Issues提交

---

**最后更新**: 2026-01-21
