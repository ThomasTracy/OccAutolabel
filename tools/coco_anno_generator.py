import os
import json
import numpy as np
from pathlib import Path
import glob
from typing import Dict, List, Tuple

def find_closest_image(timestamp: float, image_timestamps: List[float]) -> Tuple[float, float]:
    """
    在图片时间戳列表中找到最接近给定时间戳的图片
    
    Args:
        timestamp: 目标时间戳
        image_timestamps: 图片时间戳列表
        
    Returns:
        Tuple[最接近的时间戳, 时间差]
    """
    if not image_timestamps:
        return None, float('inf')
    
    # 找到最接近的时间戳
    closest_timestamp = min(image_timestamps, key=lambda x: abs(x - timestamp))
    time_diff = abs(closest_timestamp - timestamp)
    
    return closest_timestamp, time_diff

def parse_timestamp(filename: str) -> float:
    """
    从文件名中解析时间戳
    
    Args:
        filename: 文件名（可以包含路径）
        
    Returns:
        时间戳（浮点数）
    """
    # 去掉路径和扩展名，只保留文件名
    basename = Path(filename).stem
    try:
        return float(basename)
    except ValueError:
        print(f"警告: 无法解析文件名 {filename} 的时间戳")
        return None

def load_npz_annotations(npz_path: str) -> Dict:
    """
    加载npz文件中的标注信息
    
    Args:
        npz_path: npz文件路径
        
    Returns:
        包含标注信息的字典
    """
    try:
        data = np.load(npz_path)
        annotations = {}
        for key in data.files:
            annotations[key] = data[key]
        return annotations
    except Exception as e:
        print(f"错误: 无法加载npz文件 {npz_path}: {e}")
        return None

def generate_coco_annotations(root_dir: str, max_time_diff: float = 0.1) -> Dict:
    """
    生成COCO格式的标注
    
    Args:
        root_dir: 根目录路径
        max_time_diff: 最大允许的时间差（秒）
        
    Returns:
        COCO格式的标注字典
    """
    # 路径配置
    front_image_dir = os.path.join(root_dir, "images", "front")
    back_image_dir = os.path.join(root_dir, "images", "back")
    left_image_dir = os.path.join(root_dir, "images", "left")
    right_image_dir = os.path.join(root_dir, "images", "right")
    anno_dir = os.path.join(root_dir, "annotations", "gt")
    
    # 收集所有文件的时间戳
    print("正在收集文件时间戳...")
    
    # 收集标注文件时间戳
    anno_files = glob.glob(os.path.join(anno_dir, "*.npz"))
    anno_timestamps = {}
    for anno_file in anno_files:
        timestamp = parse_timestamp(anno_file)
        if timestamp is not None:
            anno_timestamps[timestamp] = anno_file
    
    # 收集前相机图片时间戳
    front_images = glob.glob(os.path.join(front_image_dir, "*.*"))
    front_timestamps = []
    front_image_map = {}
    for img_file in front_images:
        timestamp = parse_timestamp(img_file)
        if timestamp is not None:
            front_timestamps.append(timestamp)
            front_image_map[timestamp] = img_file
    
    # 收集后相机图片时间戳
    back_images = glob.glob(os.path.join(back_image_dir, "*.*"))
    back_timestamps = []
    back_image_map = {}
    for img_file in back_images:
        timestamp = parse_timestamp(img_file)
        if timestamp is not None:
            back_timestamps.append(timestamp)
            back_image_map[timestamp] = img_file
    
    # 收集左相机图片时间戳
    left_images = glob.glob(os.path.join(left_image_dir, "*.*"))
    left_timestamps = []
    left_image_map = {}
    for img_file in left_images:
        timestamp = parse_timestamp(img_file)
        if timestamp is not None:
            left_timestamps.append(timestamp)
            left_image_map[timestamp] = img_file
    
    # 收集右相机图片时间戳
    right_images = glob.glob(os.path.join(right_image_dir, "*.*"))
    right_timestamps = []
    right_image_map = {}
    for img_file in right_images:
        timestamp = parse_timestamp(img_file)
        if timestamp is not None:
            right_timestamps.append(timestamp)
            right_image_map[timestamp] = img_file
    
    print(f"找到 {len(anno_timestamps)} 个标注文件")
    print(f"找到 {len(front_timestamps)} 个前相机图片")
    print(f"找到 {len(back_timestamps)} 个后相机图片")
    print(f"找到 {len(left_timestamps)} 个左相机图片")
    print(f"找到 {len(right_timestamps)} 个右相机图片")
    
    # 初始化COCO格式数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息（根据您的实际需求修改）
    # 这里需要根据您的npz文件中的类别信息来设置
    categories = [
        {"id": 1, "name": "drive_plane"},
        {"id": 2, "name": "drive_stair"},
        {"id": 3, "name": "drive_other"},
        {"id": 4, "name": "human"},
        {"id": 5, "name": "vehicle"},
        {"id": 0, "name": "undrivable space"},
        # 添加更多类别...
    ]
    coco_data["categories"] = categories
    
    image_id = 0
    annotation_id = 0
    
    # 处理每个标注文件
    print("正在生成COCO标注...")
    
    for anno_timestamp, anno_path in anno_timestamps.items():
        # 找到最接近的前后相机图片
        front_ts, front_diff = find_closest_image(anno_timestamp, front_timestamps)
        back_ts, back_diff = find_closest_image(anno_timestamp, back_timestamps)
        left_ts, left_diff = find_closest_image(anno_timestamp, left_timestamps)
        right_ts, right_diff = find_closest_image(anno_timestamp, right_timestamps)
        
        # 检查时间差是否在允许范围内
        if front_diff > max_time_diff and back_diff > max_time_diff and left_diff > max_time_diff and right_diff > max_time_diff:
            print(f"警告: 标注 {anno_timestamp} 没有找到时间差在 {max_time_diff} 秒内的图片")
            continue
        
        # 加载标注数据
        npz_data = load_npz_annotations(anno_path)
        if npz_data is None:
            continue

        anno = {
            "id": annotation_id,
            "timestamp": int(anno_timestamp),
            "file_name": os.path.basename(anno_path)
        }
        anno_image_ids = []
        # 处理前相机图片
        if front_diff <= max_time_diff:
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(front_image_map[front_ts]),
                "width": 1920,  # 需要根据实际图片尺寸修改
                "height": 1536, # 需要根据实际图片尺寸修改
                "camera": "front",
                "timestamp": int(front_ts)
            }
            coco_data["images"].append(image_info)
            anno_image_ids.append(image_id)
            image_id += 1
        
        # 处理后相机图片
        if back_diff <= max_time_diff:
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(back_image_map[back_ts]),
                "width": 1920,  # 需要根据实际图片尺寸修改
                "height": 1536, # 需要根据实际图片尺寸修改
                "camera": "back",
                "timestamp": back_ts
            }
            coco_data["images"].append(image_info)
            anno_image_ids.append(image_id)
            image_id += 1
        
        if left_diff <= max_time_diff:
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(left_image_map[left_ts]),
                "width": 1920,  # 需要根据实际图片尺寸修改
                "height": 1536, # 需要根据实际图片尺寸修改
                "camera": "left",
                "timestamp": left_ts
            }
            coco_data["images"].append(image_info)
            anno_image_ids.append(image_id)
            image_id += 1
        
        if right_diff <= max_time_diff:
            image_info = {
                "id": image_id,
                "file_name": os.path.basename(right_image_map[right_ts]),
                "width": 1920,  # 需要根据实际图片尺寸修改
                "height": 1536, # 需要根据实际图片尺寸修改
                "camera": "right",
                "timestamp": right_ts
            }
            coco_data["images"].append(image_info)
            anno_image_ids.append(image_id)
            image_id += 1
        
        anno["image_ids"] = anno_image_ids
        coco_data["annotations"].append(anno)
        annotation_id += 1
        
    
    print(f"生成完成: {len(coco_data['images'])} 张图片, {len(coco_data['annotations'])} 个标注")
    return coco_data

def parse_annotations_from_npz(npz_data: Dict, image_id: int, start_annotation_id: int, camera: str) -> List[Dict]:
    """
    从npz数据中解析标注信息
    
    Args:
        npz_data: npz文件数据
        image_id: 图片ID
        start_annotation_id: 起始标注ID
        camera: 相机类型
        
    Returns:
        COCO格式的标注列表
    """
    annotations = []
    annotation_id = start_annotation_id
    
    # 这里需要根据您的npz文件的实际结构来解析
    # 以下是一个示例实现，您需要根据实际情况修改
    
    # 假设npz文件中包含以下键：
    # - bboxes: 边界框 [x, y, width, height]
    # - labels: 类别标签
    # - scores: 置信度（可选）
    
    if 'bboxes' in npz_data and 'labels' in npz_data:
        bboxes = npz_data['bboxes']
        labels = npz_data['labels']
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # 根据相机类型调整边界框（如果需要）
            if camera == "back":
                # 后相机可能需要特殊的边界框处理
                pass
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                "area": float(bbox[2] * bbox[3]),  # width * height
                "iscrowd": 0,
                "camera": camera
            }
            annotations.append(annotation)
            annotation_id += 1
    
    return annotations

def save_coco_annotations(coco_data: Dict, output_path: str):
    """
    保存COCO格式标注到JSON文件
    
    Args:
        coco_data: COCO格式数据
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    print(f"标注文件已保存: {output_path}")

def main():
    # 配置参数
    root_dir = "/home/jszr/Data/OCC_Autolabel/DATA/TOWN/clip_160_no_distort"  # 修改为您的根目录路径
    output_file = "/home/jszr/Data/OCC_Autolabel/DATA/TOWN/clip_160_no_distort/annotations.json"
    max_time_diff = 0.1  # 最大允许时间差（秒）
    
    # 生成COCO标注
    coco_data = generate_coco_annotations(root_dir, max_time_diff)
    
    # 保存标注文件
    save_coco_annotations(coco_data, output_file)

if __name__ == "__main__":
    main()