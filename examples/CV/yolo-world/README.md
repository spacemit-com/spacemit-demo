# YOLO-World

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
输出：输出为(1,84,2100)；84维度中0-3为未还原的左上角，右下角坐标，4-83为每个类的得分值；2100为框的数量
```

### 2.1 Python Demo

依赖安装：

```shell
sudo apt update && sudo apt install libsleef-dev 
pip install sympy==1.13.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirement.txt  --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
pip install clip-for-odlabel -i  https://mirrors.aliyun.com/pypi/simple/      
pip install spacemit-ort --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```



执行方法:

```shell
cd python
python test_yolo-world.py --classes people car bottle（用空格符隔开）
# 其他重要参数
# -model 默认为../model/yolov8s-worldv2.onnx
# --image 默认为 ../data/test.jpg
```

