#!/bin/bash

# 脚本功能：将root目录下各个clip文件夹中的occ_gt文件夹复制到目标文件夹，并以clip名称重命名

# 使用方法：
# ./pick_gt_folders.sh <root_path> <output_path>
# 示例: ./pick_gt_folders.sh /path/to/root /path/to/output

# 检查参数数量
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <root_path> <output_path>"
    echo "  root_path: 包含多个clip文件夹的根目录"
    echo "  output_path: 输出目标文件夹路径"
    exit 1
fi

ROOT_PATH="$1"
OUTPUT_PATH="$2"

# 检查root路径是否存在
if [ ! -d "$ROOT_PATH" ]; then
    echo "错误: 根目录不存在: $ROOT_PATH"
    exit 1
fi

# 创建输出目录
if [ ! -d "$OUTPUT_PATH" ]; then
    echo "创建输出目录: $OUTPUT_PATH"
    mkdir -p "$OUTPUT_PATH"
fi

echo "开始处理..."
echo "根目录: $ROOT_PATH"
echo "输出目录: $OUTPUT_PATH"
echo "----------------------------------------"

# 计数器
count=0
success=0
failed=0

# 遍历root下的所有clip文件夹
for clip_dir in "$ROOT_PATH"/*; do
    # 检查是否为目录
    if [ ! -d "$clip_dir" ]; then
        continue
    fi
    
    # 获取clip文件夹名称
    clip_name=$(basename "$clip_dir")
    
    # 检查occ_gt子文件夹是否存在
    occ_gt_path="$clip_dir/occ_gt"
    if [ ! -d "$occ_gt_path" ]; then
        echo "警告: $clip_name 中未找到 occ_gt 文件夹，跳过"
        ((failed++))
        continue
    fi
    
    # 目标路径：以clip名称命名
    target_path="$OUTPUT_PATH/$clip_name"
    
    # 复制文件夹
    echo "复制: $clip_name/occ_gt -> $clip_name"
    cp -r "$occ_gt_path" "$target_path"
    
    if [ $? -eq 0 ]; then
        ((success++))
        echo "  ✓ 成功"
    else
        ((failed++))
        echo "  ✗ 失败"
    fi
    
    ((count++))
done

echo "----------------------------------------"
echo "处理完成！"
echo "总计处理: $count 个clip文件夹"
echo "成功: $success"
echo "失败: $failed"
echo "输出目录: $OUTPUT_PATH"




# 用法
# ./pick_gt_folders.sh /path/to/root /path/to/output