# YOLOv8-pose

## 1. 模型获取

```shell
cd model
sh download_model.sh
```

执行完下载模型脚本会保存FP32的yolov8n-pose.onnx模型和INT8的yolov8n-pose.q.onnx模型

## 2. Demo 

输入输出数据说明：

```
输入：三通道图像路径
输出：输出为(1,56,2100)；56维度中0-4中心点坐标、宽、高、置信度，5-55为17个人体关键点的坐标值和置信度；2100为框的数量
```

### 2.1 Python Demo

依赖安装:

```shell
cd python
sudo apt install python3-pip python3-venv
python3 -m venv name(虚拟环境名) 
source name/bin/activate 
pip install -r requirements.txt --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

执行方法:

```shell
python test_yolov8_pose.py 
# 其他重要参数
# -model 默认为../model/yolov8n-pose.q.onnx
# --image 默认为 ../data/test.jpg
```

