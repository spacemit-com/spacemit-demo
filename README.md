# Bianbu AI Demo Zoo

## 简介

Bianbu AI Demo Zoo是基于进迭时空K系列芯片适配的AI应用示例。包含传统CV模型、LLM模型、语音处理模型，同时提供python版本和C++版本。旨在为开发者提供简单易用的模型部署教程。



## CV模型性能

性能数据不包含前后处理的时间。

|   模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) |                       Demo链接                       |
| :----------: | :-------------------: | :-----------: | :------: | :-------: | :--------------------------------------------------: |
| EfficientNet |    EfficientNet_b1    | [1,3,224,224] |   int8   |    23     | [EfficientNet](./examples/CV/efficientnet/README.md) |
|  Inception   |     Inception_v1      | [1,3,224,224] |   int8   |    18     | [Inception_v1](./examples/CV/inception_v1/README.md) |
|              |     Inception_v3      | [1,3,229,229] |   int8   |    18     | [Inception_v3](./examples/CV/inception_v3/README.md) |
|  MobileNet   |      MobileNetv2      | [1,3,224,224] |   int8   |    67     | [MobileNetv2](./examples/CV/mobilenet_v2/README.md)  |
|    ResNet    |       ResNet50        | [1,3,224,224] |   int8   |    25     |      [ResNet50](./examples/CV/resnet/README.md)      |
|    YOLOv5    |        YOLOv5n        | [1,3,640,640] |   int8   |     8     |       [yolov5](./examples/CV/yolov5/README.md)       |
|    YOLOv6    |        YOLOv6n        | [1,3,320,320] |   int8   |    54     |       [yolov6](./examples/CV/yolov6/README.md)       |
|    YOLOv8    |        YOLOv8n        | [1,3,320,320] |   int8   |    73     |       [yolov8](./examples/CV/yolov8/README.md)       |
|              |        YOLOv8n        | [1,3,192,320] |   int8   |    106     |       [yolov8](./examples/CV/yolov8/README.md)       |
|   YOLOv11    |       YOLOv11n        | [1,3,320,320] |   int8   |    35     |      [yolov11](./examples/CV/yolov11/README.md)      |
|  NanoTrack   |       NanoTrack       | [1,3,255,255] |   int8   |    55     |    [NanoTrack](./examples/CV/nanotrack/README.md)    |
|   ArcFace    | arcface_mobilefacenet | [1,3,320,320] |   int8   |    27     |      [ArcFace](./examples/CV/arcface/README.md)      |
| YOLOv5-face  |     YOLOv5n-face      | [1,3,320,320] |   int8   |    21     |  [YOLOv5-face](./examples/CV/yolov5-face/README.md)  |



## LLM模型性能

| 模型 | 参数量 | 模型格式 | 量化格式 | 存储大小 | 内存 | 推理速度（4核） |作用|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-0.5B | 0.5b | gguf | q4_0 | 336MB | 480MB | 11tokens/s | 对话 |
| Deepseek R1-1.5B | 1.5b | gguf | q4_0 | 1017MB | 1.26GB | 4.5tokens/s | 对话 |
| Qwen2.5-0.5B-f16-agv-fc | 0.5b | gguf | q4_0 | 271MB | 480MB | 11tokens/s | 函数调用 |
| Qwen2.5-0.5B-f16-elephant-fc | 0.5b | gguf | q4_0 | 271MB | 480MB | 11tokens/s | 函数调用 |
| smollm2:135m | 135m | gguf | fp16 | 271MB | 490MB | 15tokens/s | 对话 |
| smollm2-135m-f16-agv-fc | 135m | gguf | fp16 | 270MB | 480MB | 15tokens/s | 函数调用 |
| smollm2-135m-q40-agv-fc | 135m | gguf | q4_0 | 77MB | 320MB | 30tokens/s | 函数调用 |



## ASR模型性能

| 模型 | 参数量 | 模型格式 | 量化格式 | 存储大小 | 内存 | 推理速度（2核） |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| sensevoice-small（python） | 330m | onnx | 动态量化 | 229MB | 430MB | RTF=0.38 |
| sensevoice-small（c++） | 330m | onnx | 动态量化 | 229MB | 360MB | RTF=0.3 |



## TTS模型性能

| 模型 | 模型格式 | 量化格式 | 存储大小 | 内存 | 推理速度（4核） |
| :---: | :---: | :---: | :---: | :---: | :---: |
| melotts | onnx | 动态量化 | 74MB | 100MB | 2<RTF<4 |
| matchtts | onnx | model-steps-3动态量化 | 73MB | 300MB | 0.64<RTF<0.75 |



