# FCN

## 1. 模型和数据获取

### 1.1 模型获取

```shell
cd model
sh download_model.sh
```

执行完后会在当前目录下载两个模型fcn_r50.onnx和fcn_r50.q.onnx。

fcn_r50.onnx为FP32模型。

fcn_r50.q.onnx为INT8模型。





## 2. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：(1,512,512),输出为单通道灰度图
```



### 2.1 python demo

依赖安装：

```shell
pip install opencv-python==4.11.0 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install spacemit-ort==1.2.2 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

执行流程:

```shell
cd python
python test_fcn.py

# 参数说明
#--model 模型具体路径(默认为../model/fcn_r50.q.onnx)  
#--image 测试图片路径（默认为../data/test_unet.jpg）

```

结果保存为result.jpg。

### 2.2 c++ demo

```shell
cd cpp
mkdir build
cd build
cmake ..
make -j8
./fcn_demo --model onnx模型路径 --image 图片路径

```

结果保存为result.jpg。
