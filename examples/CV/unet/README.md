# Unet

## 1. 模型和数据获取

### 1.1 模型获取

```shell
cd model
sh download_model.sh
```

执行完后会在当前目录下载两个模型unet-s5.onnx和unet-s5.q.onnx。

unet-s5.onnx为FP32模型。

unet-s5.q.onnx为INT8模型。





## 2. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：(1,512,512),输出为单通道灰度图
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

执行方法：

```shell
python test_unet.py

# 参数说明
#--model 模型具体路径(默认为../model/unet-s5.q.onnx)  
#--image 测试图片路径（默认为../data/test_unet.jpg）

```

结果保存为result.jpg。

### 2.2 c++ demo

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
./unet_demo --model onnx模型路径 --image 图片路径

```

结果保存为result.jpg。
