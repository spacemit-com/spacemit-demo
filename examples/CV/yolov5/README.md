# YOLOv5

## 1. 模型和数据获取

### 1.1 模型获取

```shell
cd model
sh download_model.sh
```

执行完后会在当前目录下载两个模型yolov5_n.onnx和yolov5_n.q.onnx。

yolov5_n.onnx为FP32模型。

yolov5_n.q.onnx为INT8模型。



### 1.2 数据获取

```shell
cd data
sh download_data.sh
```

执行完后会在当前目录下载测试图片test.jpg和标签文件label.txt



## 2. 模型量化

如果用1.1中的INT8模型，可跳过该步骤。

**Note**:注意请在x86平台就行模型量化



### 2.1 量化工具安装

量化浮点模型需要安装我们的Xquant量化工具，安装步骤如下：

```shell
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip
sudo apt install python3-virtualenv
pip install xquant==1.2.2 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com  --extra-index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

```



### 2.2 转换浮点模型为量化模型

[Calibration数据下载]: https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/Datasets/Coco/Coco.tar.gz



```shell
tar -xzvf Coco.tar.gz 
virtualenv xquant_env
source xuqant_env/bin/activate
cd model 
sh download_quant_config.sh

```

执行完后会下载xquant_config.json,修改xquant_config.json文件中以下参数:

"onnx_model": 浮点onnx模型的路径

"data_list_path”: 具体的calib_img_list.txt路径(在解压的Coco.tar.gz的目录中)

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
输出：输出有两个;outputs[0]维度为(number_dets,5),0-3为未还原的左上角，右下角坐标，4为分类得分值;outputs[1]维度为(number_dets)，保存的是每个输出对应的类别
```



### 3.1 python demo

依赖安装：

```shell
pip install opencv-python==4.11.0 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install spacemit-ort==1.2.2 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

执行流程:

```shell
cd python
python test_yolov5.py

# 参数说明
#--model 模型具体路径(默认为../model/yolov5_n.q.onnx)  
#--image 测试图片路径（默认为../data/test.jpg）
#--use-camera 是否使用摄像头（需要摄像头时带上此参数）
#--conf-threshold 置信度阈值（可选）

# 如需迁移代码，注意utils.py中标签文件默认为../data/label.txt.
```

结果保存为result.jpg

### 3.2 c++ demo

```shell
cd cpp
mkdir build
cd build
cmake ..
make -j8
./yolov5_demo --model onnx模型路径 --image 图片路径
# 注意main.cc中labelFilePath的路径是否正确，默认是../../data/label.txt(要以build目录为基准路径)
```

结果保存为result.jpg
