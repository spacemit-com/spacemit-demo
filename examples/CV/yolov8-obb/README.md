# YOLOv8-OBB

## 1. 模型和数据获取

### 1.1 模型获取

```shell
cd model
sh download_model.sh
```

下载模型保存为yolov8n-obb.onnx。

### 1.2 数据获取

```shell
cd data
sh download_data.sh
```

执行完后会在当前目录下载测试图片obb_demo.jpg和标签文件label.txt。

## 2. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：输出维度为(1,20,2100),其中20维度前4表示坐标框的（中心点坐标，宽，高），4-19表示15个类的概率，20表示角度。2100表示框得数量。
```

### 2.1 python demo


依赖安装：

```shell
cd python
sudo apt install python3-pip python3-venv
python3 -m venv name(虚拟环境名) 
source name/bin/activate 
pip install -r requirements.txt --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

执行方法:

```shell
python test_yolov8-obb.py

# 参数说明
#--model 模型路径，默认为../model/yolov8n-obb.onnx
#--image 输入图像路径，默认为../data/obb_demo.jpg
#--use-camera 使用摄像头推理(可选)
#--conf-threshold 置信度阈值，默认为0.25
#--iou-threshold IoU阈值，默认为0.45

```
结果保存为result.jpg