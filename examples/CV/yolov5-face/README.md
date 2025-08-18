# YOLOv5-face

## 1.模型数据获取

```shell
cd data/
sh download_data.sh 
cd ../model
sh download_model.sh
```

  

## 2. Demo

输入输出数据说明：

```
输入:三通道图像路径
输出:左上角和右下角的坐标
```



### 2.1 python demo

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
python  test_yolov5-face.py
# 重要参数
# --model 模型路径，默认为../model/yolov5n-face_320_cut.q.onnx
# --image 图片路径，默认为../data/test.jpg
# --use-camera 带上此参数时启用摄像头
```

执行结果：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="images/1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> 图1 人脸检测结果 </div>
</center>