# MobileSam1

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
输入：encoder以及decoder模型的路径，测试图像（此图像是包含apha信息在内的四通道图像），提示框信息，掩膜输入信息（默认不用输入）
输出：根据提示框信息输出包含有分割掩膜叠加的测试图像
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
```

执行方法:

```shell
python mobilesam1_onnx_infer.py
```

