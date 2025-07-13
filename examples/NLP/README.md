# Bianbu AI Demo Zoo

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-SpacemiT%20K1-green.svg)]()

本项目是SpacemiT K1开发板上的AI功能演示合集，包含音频处理、语音识别、大语言模型、文本转语音、视觉语言模型等多种AI应用示例。

## 🚀 快速开始

### 克隆代码仓库

```bash
git clone https://gitee.com/bianbu/spacemit-demo.git
cd spacemit_demo/examples/NLP
```

### 安装基础依赖

```bash
sudo apt update
sudo apt install libportaudio2 libopenblas-dev python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## 📚 功能列表

### 🎙️ 音频采集功能

#### USB麦克风音频采集
- **文件**: `02_capture_audio.py`
- **功能**: 通过USB麦克风进行音频采集
- **特点**: 
  - 单次采集最长4秒
  - 输出PCM格式音频文件
  - 支持实时录音控制

**使用方法**：
```bash
python 02_capture_audio.py
```

#### 环形麦克风音频采集
- **文件**: `04_asr_demo_db.py`
- **功能**: 使用环形麦克风阵列进行音频采集
- **特点**:
  - 单声道音频采集
  - 最大录音时长5秒
  - 支持麦克风阵列降噪

**使用方法**：
```bash
python 04_asr_demo_db.py
```

### 🤖 AI功能体验

#### 1. 语音活动检测 (VAD)
- **文件**: `01_search_device.py`
- **功能**: 自动检测语音活动，控制录音启停
- **特点**:
  - 实时语音检测
  - 自动录音控制
  - 支持设备枚举

**准备工作**：
```bash
# 下载VAD模型
wget -O ~/.cache/sensevoice.tar.gz https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz 
tar -xzf ~/.cache/sensevoice.tar.gz -C ~/.cache
rm ~/.cache/sensevoice.tar.gz

# 安装依赖
sudo apt install onnxruntime python3-spacemit-ort python3-numpy python3-pyaudio
```

**检测录音设备**：
```bash
# 方法1：使用arecord
arecord -l

# 方法2：运行脚本
python3 01_search_device.py
```

#### 2. 语音转文本 (ASR)
- **文件**: `03_asr_demo.py`
- **功能**: 将语音实时转换为文本
- **特点**:
  - 集成VAD功能
  - 静音自动停止
  - 支持参数配置

**使用方法**：
```bash
python 03_asr_demo.py
```

**主要参数**：
| 参数名称 | 说明 | 默认值 |
|---------|------|--------|
| `sld` | 静音长度阈值（秒） | 可配置 |
| `max_time` | 最长录音时间（秒） | 可配置 |
| `channels` | 音频通道数 | 1 |
| `rate` | 采样率（Hz） | 16000/48000 |
| `device_index` | 输入设备索引 | 可配置 |

#### 3. 大语言模型 (LLM)
- **工具**: Ollama
- **功能**: 本地部署和运行大语言模型
- **支持模型**: LLaMA, DeepSeek, Qwen等

**安装Ollama**：
```bash
sudo apt install spacemit-ollama-toolkit
```

**验证安装**：
```bash
ollama list
```

**使用示例**：
```bash
# 运行Qwen模型
ollama run qwen3:0.6b
```

**手动制作模型**：
```bash
# 下载模型文件
wget https://modelscope.cn/models/second-state/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/Qwen2.5-0.5B-Instruct-Q4_0.gguf -P ~/
wget https://archive.spacemit.com/spacemit-ai/modelfile/qwen2.5:0.5b.modelfile -P ~/

# 创建模型
ollama create qwen2.5:0.5b -f qwen2.5:0.5b.modelfile

# 启动模型
ollama run qwen2.5:0.5b
```

#### 4. 语音输入大模型输出
- **文件**: `06_asr_llm_demo.py`
- **功能**: 语音识别 + 大语言模型推理的完整流程
- **特点**:
  - 完全离线运行
  - 语音到文本到智能回复
  - 集成VAD功能

**一键部署**：
```bash
sudo apt install asr-llm
voice
```

**手动运行**：
```bash
python 06_asr_llm_demo.py
```

**工作流程**：
1. 自动录音并进行语音识别
2. 将识别文本传递给大语言模型
3. 返回智能回复结果

#### 5. 文本转语音 (TTS)
- **文件**: `07_tts_demo.py`
- **功能**: 将文本转换为语音输出
- **特点**:
  - 支持多种播放设备
  - 高质量语音合成
  - 实时文本转语音

**一键部署**：
```bash
sudo apt install asr-llm-tts
tts
```

**检测播放设备**：
```bash
# 方法1：使用aplay
aplay -l

# 方法2：使用pactl
sudo apt install pulseaudio-utils
pactl list short sinks

# 设置默认播放设备
pactl set-default-sink [设备名称]
```

**手动运行**：
```bash
python 07_tts_demo.py
```

#### 6. 函数调用 (Function Calling)
- **文件**: `05_llm_demo.py`
- **功能**: 大语言模型自动选择和调用函数
- **特点**:
  - 智能意图解析
  - 自动函数选择
  - 结构化响应

**模型准备**：
```bash
# 下载函数调用专用模型
wget http://archive.spacemit.com/spacemit-ai/gguf/qwen2.5-0.5b-fc-q4_0.gguf -P ~/
wget http://archive.spacemit.com/spacemit-ai/modelfile/qwen2.5-0.5b-fc.modelfile -P ~/

# 创建模型
ollama create qwen2.5-0.5b-fc -f qwen2.5-0.5b-fc.modelfile
```

**使用方法**：
```bash
python 05_llm_demo.py
```

#### 7. 视觉语言模型 (VLM)
- **文件**: `08_vision_demo.py`
- **功能**: 图像理解和文本生成
- **模型**: SmolVLM
- **特点**:
  - 图像+文本多模态输入
  - 本地离线推理
  - 自然语言图像描述

**模型准备**：
```bash
# 下载SmolVLM模型
wget https://archive.spacemit.com/spacemit-ai/gguf/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
wget https://archive.spacemit.com/spacemit-ai/gguf/SmolVLM-256M-Instruct-f16.gguf
wget https://archive.spacemit.com/spacemit-ai/modelfile/smolvlm.modelfile

# 创建模型
ollama create smolvlm:256m -f smolvlm.modelfile
```

**使用方法**：
```bash
python 08_vision_demo.py --image=bus.jpg --stream=True --prompt="describe this image"
```

## 🛠️ 系统要求

- **硬件**: SpacemiT K1 开发板
- **系统**: Bianbu Linux
- **固件版本**: ≥ 2.2 (推荐最新版本)
- **Python**: 3.8+

## 📂 项目结构

```
spacemit_demo/examples/NLP/
├── 01_search_device.py          # 设备检测
├── 02_capture_audio.py          # USB麦克风采集
├── 03_asr_demo.py              # 语音转文本
├── 04_asr_demo_db.py           # 环形麦克风采集
├── 05_llm_demo.py              # 函数调用
├── 06_asr_llm_demo.py          # 语音+大模型
├── 07_tts_demo.py              # 文本转语音
├── 08_vision_demo.py           # 视觉语言模型
├── requirements.txt            # Python依赖
└── README.md                   # 说明文档
```

## 🔧 故障排除

### 音频设备问题
1. 确认麦克风/扬声器已正确连接
2. 检查设备权限设置
3. 使用 `arecord -l` 和 `aplay -l` 确认设备识别

### 模型下载问题
1. 检查网络连接
2. 确认磁盘空间充足
3. 使用wget重新下载模型文件

### 依赖安装问题
1. 更新系统包列表：`sudo apt update`
2. 检查Python版本兼容性
3. 使用虚拟环境隔离依赖

## 📞 技术支持

- **官方文档**: [SpacemiT K1 开发指南](https://developer.spacemit.com)
- **社区论坛**: [SpacemiT 开发者社区](https://community.spacemit.com)
- **Issue反馈**: [Gitee Issues](Issues · Bianbu/spacemit-demo - Gitee.com)

## 📄 开源协议

本项目采用 Apache 2.0 开源协议，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目！

1. Fork 本仓库
2. 创建特性分支
3. 提交代码变更
4. 推送到分支
5. 创建Pull Request

---

**Copyright © 2024 SpacemiT. All rights reserved.**
