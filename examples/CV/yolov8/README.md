# YOLOv8

## 1. 模型和数据获取

### 1.1 模型导出

官方onnx模型导出，可以参考[Ultralytics官方导出文档](https://docs.ultralytics.com/zh/modes/export/#introduction)。

### 1.2 模型获取

```shell
cd model
sh download_model.sh
```

下载模型保存为yolov8n_192x320.q.onnx,yolov8n_320x320.q.onnx。



### 1.3 数据获取

```shell
cd data
sh download_data.sh
```

执行完后会在当前目录下载测试图片test.jpg和标签文件label.txt。



## 2. 模型量化

如果用1.1中的INT8模型，可跳过该步骤。

**Note**:注意请在x86平台就行模型量化



### 2.1 量化工具安装

量化浮点模型需要安装我们的Xquant量化工具，安装步骤如下：

```shell
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip
sudo apt install python3-virtualenv
pip install xquant -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  --extra-index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

```



### 2.2 转换浮点模型为量化模型

[Calibration数据下载](https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/Datasets/Coco/Coco.tar.gz)



```shell
tar -xzvf Coco.tar.gz 
virtualenv xquant_env
source xquant_env/bin/activate
cd model 
sh download_quant_config.sh

```

执行完后会下载xquant_config.json和yolov8_preprocess.py,修改xquant_config.json文件中以下参数:

"onnx_model": 浮点onnx模型的路径

"data_list_path”: 具体的calib_img_list.txt路径(在解压的Coco.tar.gz的目录中)

"preprocess_file":自定义预处理文件(yolov8_preprocess.py)和指定函数

"working_dir": 为输出文件保存路径(默认temp，可选)

执行下面命令：

```shell
python -m xuqant --config xquant_config.json
```

最终量化模型保存在working_dir目录下，后缀为.q.onnx。



## 3. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：输出为(1,84,2100)；84维度中0-3为未还原的左上角，右下角坐标，4-83为每个类的得分值；2100为框的数量
```



### 3.1 python demo

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
python test_yolov8.py

# 参数说明
#--model 模型具体路径(默认为../model/yolov8n.q.onnx)  
#--image 测试图片路径（默认为../data/test.jpg）
#--use-camera 是否使用摄像头（需要摄像头时带上此参数）
#--conf-threshold 置信度阈值(可选)
#--iou-threshold IOU阈值(可选)

# 如需迁移代码，注意utils.py中标签文件默认为../data/label.txt.
```

结果保存为result.jpg

### 3.2 c++ demo

依赖安装：

```shell
sudo apt install libopencv-dev
```

执行方法：

```shell
cd cpp
mkdir build
cd build
cmake ..
make -j8
./yolov8_demo --model onnx模型路径 --image 图片路径
# 注意main.cc中labelFilePath的路径是否正确，默认是../../data/label.txt(要以build目录为基准路径)
```

结果保存为result.jpg。
