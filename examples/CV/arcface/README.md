# ArcFace

## 1. 模型获取

```
cd model
download_model.sh
```



## 2. Demo

输入输出数据说明：

```
输入:需要对比的两张人脸图片
输出:两个人脸特征值，大小为[1,128]
```



### 2.1 Python Demo

依赖安装：

```shell
cd python
sudo apt install python3-pip python3-venv
python3 -m venv name(虚拟环境名) 
source name/bin/activate 
pip install -r requirement.txt --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

执行方法：

```shell
python test_arcface.py
# 重要参数
# --image1 对比图片1，默认为../data/face0.png
# --image2 对比图片2，默认为../data/face1.png
```

执行结果：

```
相似度:17.74%.  
```

