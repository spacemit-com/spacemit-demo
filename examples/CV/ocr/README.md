# OCR

## 1. 模型和数据获取

### 1.1 模型导出

Paddle官方onnx模型导出，可以参考[PaddleOCR官方导出文档](https://www.paddleocr.ai/main/version3.x/deployment/obtaining_onnx_models.html)。

### 1.2 模型获取

```shell
cd model
sh download_model.sh
```

下载模型保存为ppocr3_det_fixed.onnx，ppocr_rec.onnx。



### 1.3 数据获取

```shell
cd data
sh download_data.sh
```

执行完后会在当前目录下载测试图片sign.jpg和字典文件rec_word_dict.txt





## 2. Demo

输入输出数据说明：

```
输入：三通道图像路径
输出：json，包含目标得分，目标中心点位置，OCR识别结果。
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

执行方法:

```shell
python test_ocr.py

# 参数说明
#--model_det 文字检测模型路径，默认为../model/ppocr3_det_fixed.onnx
#--model_rec 文字识别模型路径，默认为../model/ppocr_rec.onnx
#--dict_path 字典文件路径，默认为../data/rec_word_dict.txt
#--use-camera 使用摄像头推理(可选)
#--save-warp-img 保存结果图片(可选)

```

