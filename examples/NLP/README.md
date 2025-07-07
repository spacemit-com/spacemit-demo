## 安装环境
```bash
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

## ASR demo
```bash
python asr_demo.py
```
点击enter输入语音即可语音转文字


## ASR LLM demo
我们提供了asr-llm的deb包
```bash
sudo apt update
sudo apt install asr-llm
```
详细说明请看 /usr/share/asr-llm/README.md 文件

本仓库直接使用：
```bash
python asr_llm_demo.py
```
终端输入voice即可启动程序，点击enter输入语音即可语音转文字，大模型会先判断你的语音是否有function call指令，如果有即会去调用对应函数代码（函数表在functions.py文件里面），否则正常输出聊天文本


## LLM demo
```bash
python llm_demo.py
```
启动程序根据提示输入信息即可与大模型正常对话，如需修改模型：
```bash
vim llm_demo.py
```
修改代码里面的通用模型名字llm_model_path以及微调模型名字fc_model_path即可


## ASR LLM TTS demo
这是一个端到端语音项目。同样的，我们提供了asr-llm-tts的deb包
```bash
sudo apt update
sudo apt install asr-llm-tts
```
安装完以后需要重启计算机以启动程序服务，终端输入tts即可启动进行对话
详情请看/usr/share/asr-llm-tts/README.md文件


## TTS demo
```bash
python tts_demo
```
根据提示输入文本即可将文本输出为语音