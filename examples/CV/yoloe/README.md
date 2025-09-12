# YOLOE

## 1. 模型数据获取

```shell
cd data
sh download_data.sh
cd ../model
sh download_model.sh
```

## 2. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：输出为(1,84,2100)；84维度中0-3为未还原的中心点坐标，宽，高;4-83为每个类的得分值；2100为框的数量
```

### 2.1 Python Demo

依赖安装：

```shell
sudo apt update && sudo apt install libsleef-dev 
cd python
sudo apt install python3-pip python3-venv
python3 -m venv name(虚拟环境名) 
source name/bin/activate 
pip install sympy==1.13.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirement.txt  --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install clip-for-odlabel -i  https://mirrors.aliyun.com/pypi/simple/        
```



执行方法:

```shell
python yoloe_onnx_infer.py --mobile_clip_onnx_path ../model/mobileclip.q.onnx --yoloe_onnx_path ../model/yoloe_v8s_det.q.onnx --img_path ../data/bus.jpg --text_prompt "bus,person"

```
### 2.2 c++ Demo


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
./yoloe_demo --mobileclip_model ../../model/mobileclip.q.onnx --yoloe_model ../../model/yoloe_v8s_det.q.onnx  --image ../../data/bus.jpg --text bus,person

