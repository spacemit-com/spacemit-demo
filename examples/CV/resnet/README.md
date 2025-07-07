# ResNet

## 1. 模型和数据获取

### 1.1 模型获取

```shell
cd model
sh download_model.sh
```

执行完后会在当前目录下载两个模型resnet50.onnx和resnet50.q.onnx。

resnet50.onnx为FP32模型。

resnet50.q.onnx为INT8模型。



### 1.2 数据获取

```shell
cd data
sh download_data.sh
```

执行完后会在当前目录下载测试图片kitten.jpg和标签文件label.txt



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

[Calibration数据下载]: https://archive.spacemit.com/spacemit-ai/BRDK/Model_Zoo/Datasets/ImageNet/Imagenet.tar.gz



```shell
tar -xzvf Imagenet.tar.gz 
virtualenv xquant_env
source xuqant_env/bin/activate
cd model 
sh download_quant_config.sh

```

执行完后会下载xquant_config.json,修改xquant_config.json文件中以下参数:

"onnx_model": 浮点onnx模型的路径

"data_list_path”: 具体的calib_img_list.txt路径(在解压的Imagenet.tar.gz的目录中)

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
输出：(1,1000)的numpy，代表着每个类别的逻辑值
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
python test_resnet.py

# 参数说明
#--model 模型具体路径(默认为../model/resnet50.q.onnx)  
#--image 测试图片路径（默认为../data/kitten.jpg）

# 如需迁移代码，注意标签文件默认为../data/label.txt.
```



### 3.2 c++ demo

```shell
cd cpp
mkdir build
cd build
cmake ..
make -j8
./resnet_demo --model onnx模型路径 --image 图片路径
# 注意resnet.cc中label_file_path的路径是否正确，默认是../../data/label.txt(要以build目录为基准路径)
```

