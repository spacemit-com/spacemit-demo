from ollama import chat
import json
import webbrowser

class LLMModel:
    def __init__(self, llm_model_path='qwen2.5:0.5b', stream=True): # 你可以修改 llm_model_path 为自己的 ollama 通用大模型
        self._model_path = llm_model_path
        self._stream = stream
        pass

    # 获取聊天流的函数
    def get_chat_stream(self, text, messages):
        messages.append({"role": "user", "content": text})
        stream = chat(
            model=self._model_path,
            messages=messages,
            stream=self._stream
        )
        return stream

    # 处理函数调用的主逻辑
    def generate(self, text):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "你是一个问答助手，正常回答用户问题。\n"
                )
            }
        ] 
        
        # 获取聊天流
        stream = self.get_chat_stream(text, self.messages)

        # 处理聊天流中的每一部分
        for chunk in stream:
            content = chunk['message']['content']
            yield content



