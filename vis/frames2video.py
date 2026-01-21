#!/usr/bin/env python3
"""
将文件夹中的所有图片生成视频
支持两张图片拼接，生成视频
"""

import os
import cv2
import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple


def get_image_files(folder_path: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[str]:
    """
    获取文件夹中所有图片文件
    
    Args:
        folder_path: 图片文件夹路径
        extensions: 支持的图片格式扩展名
        
    Returns:
        排序后的图片文件路径列表
    """
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            image_files.append(os.path.join(folder_path, file))
    
    # 按文件名排序
    image_files.sort()
    return image_files


def concatenate_images_horizontal(img1: np.ndarray, img2: np.ndarray, timestamp: str = None) -> np.ndarray:
    """
    将两张图片水平拼接（左右拼接）
    
    Args:
        img1: 左侧图片
        img2: 右侧图片
        timestamp: 时间戳或帧信息（可选）
        
    Returns:
        拼接后的图片
    """
    # 获取两张图片的高度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 使用较大的高度，对较小的图片进行填充
    max_height = max(h1, h2)
    
    # 调整图片高度
    if h1 < max_height:
        img1 = cv2.resize(img1, (w1, max_height))
    if h2 < max_height:
        img2 = cv2.resize(img2, (w2, max_height))
    
    # 复制图像以避免修改原图
    img1 = img1.copy()
    img2 = img2.copy()
    
    # 在左侧图片上添加 "before" 文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_color = (0, 255, 0)  # 绿色
    
    # 获取文字尺寸
    text1 = "before"
    (text_width1, text_height1), baseline1 = cv2.getTextSize(text1, font, font_scale, font_thickness)
    
    # 计算文字位置（左上角，留一些边距）
    text_x1 = 20
    text_y1 = text_height1 + 30
    
    # 添加黑色背景使文字更清晰
    cv2.rectangle(img1, (text_x1 - 10, text_y1 - text_height1 - 10), 
                  (text_x1 + text_width1 + 10, text_y1 + baseline1 + 10), 
                  (0, 0, 0), -1)
    
    # 在左侧图片上写文字
    cv2.putText(img1, text1, (text_x1, text_y1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # 在右侧图片上添加 "after" 文字
    text2 = "after"
    (text_width2, text_height2), baseline2 = cv2.getTextSize(text2, font, font_scale, font_thickness)
    
    # 计算右上角位置
    h2_final, w2_final = img2.shape[:2]
    text_x2 = w2_final - text_width2 - 20
    text_y2 = text_height2 + 30
    
    # 添加黑色背景使文字更清晰
    cv2.rectangle(img2, (text_x2 - 10, text_y2 - text_height2 - 10), 
                  (text_x2 + text_width2 + 10, text_y2 + baseline2 + 10), 
                  (0, 0, 0), -1)
    
    # 在右侧图片上写文字
    cv2.putText(img2, text2, (text_x2, text_y2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # 水平拼接
    concatenated = np.hstack((img1, img2))
    
    # 如果提供了timestamp，在拼接图片的顶部中央显示
    if timestamp is not None:
        concat_h, concat_w = concatenated.shape[:2]
        
        # 设置timestamp文字样式
        timestamp_font = cv2.FONT_HERSHEY_SIMPLEX
        timestamp_scale = 1.2
        timestamp_thickness = 3
        timestamp_color = (0, 255, 255)  # 黄色
        
        # 获取文字尺寸
        (ts_width, ts_height), ts_baseline = cv2.getTextSize(
            timestamp, timestamp_font, timestamp_scale, timestamp_thickness
        )
        
        # 计算文字位置（顶部中央）
        ts_x = (concat_w - ts_width) // 2
        ts_y = ts_height + 20
        
        # 添加黑色背景使文字更清晰
        cv2.rectangle(concatenated, 
                      (ts_x - 15, ts_y - ts_height - 15), 
                      (ts_x + ts_width + 15, ts_y + ts_baseline + 15), 
                      (0, 0, 0), -1)
        
        # 在顶部写timestamp
        cv2.putText(concatenated, timestamp, (ts_x, ts_y), 
                    timestamp_font, timestamp_scale, timestamp_color, 
                    timestamp_thickness, cv2.LINE_AA)
    
    return concatenated


def frames_to_video_dual(folder1: str, folder2: str, output_path: str, fps: int = 30, codec: str = 'mp4v'):
    """
    将两个文件夹中的图片左右拼接后转换为视频
    
    Args:
        folder1: 左侧图片文件夹路径
        folder2: 右侧图片文件夹路径
        output_path: 输出视频文件路径
        fps: 视频帧率（默认30fps）
        codec: 视频编码格式（默认'mp4v'）
    """
    # 获取两个文件夹的所有图片文件
    image_files1 = get_image_files(folder1)
    image_files2 = get_image_files(folder2)
    
    if not image_files1:
        print(f"错误：在文件夹 {folder1} 中没有找到图片文件")
        return
    
    if not image_files2:
        print(f"错误：在文件夹 {folder2} 中没有找到图片文件")
        return
    
    # 取两个文件夹中图片数量的较小值
    min_count = min(len(image_files1), len(image_files2))
    if len(image_files1) != len(image_files2):
        print(f"警告：两个文件夹中的图片数量不一致（{len(image_files1)} vs {len(image_files2)}），将使用前 {min_count} 张图片")
    
    print(f"找到 {min_count} 对图片")
    
    # 读取第一对图片并拼接，获取输出尺寸
    first_frame1 = cv2.imread(image_files1[0])
    first_frame2 = cv2.imread(image_files2[0])
    
    if first_frame1 is None:
        print(f"错误：无法读取图片 {image_files1[0]}")
        return
    if first_frame2 is None:
        print(f"错误：无法读取图片 {image_files2[0]}")
        return
    
    # 拼接第一帧获取尺寸（带timestamp以获取正确尺寸）
    first_filename = os.path.splitext(os.path.basename(image_files1[0]))[0]
    first_concatenated = concatenate_images_horizontal(first_frame1, first_frame2, first_filename)
    height, width, _ = first_concatenated.shape
    print(f"拼接后视频尺寸: {width}x{height}")
    print(f"帧率: {fps} fps")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 {output_path}")
        return
    
    # 逐帧写入视频
    print("开始生成拼接视频...")
    for i in range(min_count):
        frame1 = cv2.imread(image_files1[i])
        frame2 = cv2.imread(image_files2[i])
        
        if frame1 is None:
            print(f"警告：跳过无法读取的图片 {image_files1[i]}")
            continue
        if frame2 is None:
            print(f"警告：跳过无法读取的图片 {image_files2[i]}")
            continue
        
        # 获取当前图片文件名作为timestamp（去掉后缀）
        filename = os.path.splitext(os.path.basename(image_files1[i]))[0]
        
        # 拼接图片
        concatenated_frame = concatenate_images_horizontal(frame1, frame2, filename)
        
        # 确保所有帧尺寸一致
        if concatenated_frame.shape[:2] != (height, width):
            concatenated_frame = cv2.resize(concatenated_frame, (width, height))
        
        video_writer.write(concatenated_frame)
        
        # 显示进度
        if (i + 1) % 10 == 0 or i == min_count - 1:
            print(f"进度: {i + 1}/{min_count}")
    
    # 释放资源
    video_writer.release()
    print(f"拼接视频生成完成: {output_path}")


def frames_to_video(input_folder: str, output_path: str, fps: int = 30, codec: str = 'mp4v'):
    """
    将文件夹中的图片序列转换为视频
    
    Args:
        input_folder: 输入图片文件夹路径
        output_path: 输出视频文件路径
        fps: 视频帧率（默认30fps）
        codec: 视频编码格式（默认'mp4v'）
    """
    # 获取所有图片文件
    image_files = get_image_files(input_folder)
    
    if not image_files:
        print(f"错误：在文件夹 {input_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"错误：无法读取图片 {image_files[0]}")
        return
    
    height, width, _ = first_frame.shape
    print(f"视频尺寸: {width}x{height}")
    print(f"帧率: {fps} fps")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 {output_path}")
        return
    
    # 逐帧写入视频
    print("开始生成视频...")
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告：跳过无法读取的图片 {img_path}")
            continue
        
        # 确保所有帧尺寸一致
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        # 显示进度
        if (i + 1) % 10 == 0 or i == len(image_files) - 1:
            print(f"进度: {i + 1}/{len(image_files)}")
    
    # 释放资源
    video_writer.release()
    print(f"视频生成完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='将文件夹中的图片序列转换为视频')
    parser.add_argument('input_folder', type=str, help='输入图片文件夹路径（或第一个文件夹，当使用--dual模式时）')
    parser.add_argument('--dual', type=str, default=None, help='第二个图片文件夹路径，用于左右拼接模式')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出视频文件路径（默认为output.mp4）')
    parser.add_argument('-f', '--fps', type=int, default=30, help='视频帧率（默认30）')
    parser.add_argument('-c', '--codec', type=str, default='mp4v', 
                        choices=['mp4v', 'XVID', 'H264', 'X264'],
                        help='视频编码格式（默认mp4v）')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.isdir(args.input_folder):
        print(f"错误：输入文件夹 {args.input_folder} 不存在")
        return
    
    # 如果是双文件夹模式
    if args.dual:
        if not os.path.isdir(args.dual):
            print(f"错误：第二个文件夹 {args.dual} 不存在")
            return
        
        # 设置默认输出路径
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.input_folder), 'output_dual.mp4')
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 生成拼接视频
        frames_to_video_dual(args.input_folder, args.dual, args.output, args.fps, args.codec)
    else:
        # 设置默认输出路径
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.input_folder), 'output.mp4')
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 生成视频
        frames_to_video(args.input_folder, args.output, args.fps, args.codec)


if __name__ == '__main__':

    # # 基本使用 - 单个文件夹
    # python tools/frames2video.py /path/to/images

    # # 指定输出路径和帧率
    # python tools/frames2video.py /path/to/images -o output_video.mp4 -f 25

    # # 使用不同编码
    # python tools/frames2video.py /path/to/images -c H264

    # # 双文件夹拼接模式 - 左右拼接两个文件夹的图片
    # python tools/frames2video.py /path/to/folder1 --dual /path/to/folder2
    
    # # 双文件夹拼接模式，指定输出路径
    # python tools/frames2video.py /path/to/folder1 --dual /path/to/folder2 -o concatenated.mp4 -f 25

    main()
