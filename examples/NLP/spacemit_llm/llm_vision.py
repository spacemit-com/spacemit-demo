from ollama import chat
import json

class VisionModel:
    def __init__(self, vision_model_path='smolvlm:256m', stream=True): # 你可以修改 llm_model_path 为自己的 ollama 通用大模型
        self._model_path = vision_model_path
        self._stream = stream
        pass

    # 获取聊天流的函数
    def get_chat_stream(self, text, messages, b64):
        messages.append({"role": "user", "content": text, "images": [b64]})
        stream = chat(
            model=self._model_path,
            messages=messages,
            stream=self._stream
        )
        return stream

    # 处理函数调用的主逻辑
    def generate(self, text, b64):
        self.b64 = b64
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual assistant. Describe images clearly and answer questions based on visual content.\n"
                )
            }
        ] 
        
        # 获取聊天流
        stream = self.get_chat_stream(text, self.messages, self.b64)

        # 处理聊天流中的每一部分
        for chunk in stream:
            content = chunk['message']['content']
            yield content



