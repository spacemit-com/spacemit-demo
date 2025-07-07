#!/bin/bash

# 检查是否提供了参数
if [ $# -ne 1 ]; then
    echo "用法: $0 [320*320|192*320]"
    exit 1
fi

# 根据参数选择下载链接
case $1 in
    320)
        URL="https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/YOLOv8/yolov8n.q.onnx"
        OUTPUT="./yolov8n.q.onnx"
        ;;
    192)
        URL="https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/CV/YOLOv8/yolov8n_192x320.q.onnx"
        OUTPUT="./yolov8n_192x320.q.onnx"
        ;;
    *)
        echo "错误: 无效的参数 '$1'"
        echo "支持的参数: 320*320, 192*320"
        exit 1
        ;;
esac

# 检查是否安装了 wget
if ! command -v wget &> /dev/null; then
    echo "错误: 需要安装 wget 才能下载文件"
    exit 1
fi

# 下载模型
echo "开始下载 $OUTPUT ..."
wget -O "$OUTPUT" "$URL"

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "下载完成: $OUTPUT"
else
    echo "下载失败"
    exit 1
fi