import json
from ollama import chat

class FCModel:
    def __init__(self, fc_model_path='qwen2.5-0.5b-fc'): # 你可以修改 fc_model_path 为自己微调后的 ollama 模型
        self._model_path = fc_model_path

    def get_chat(self, text):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Return the corresponding function content in json format based on the user's input. If there is no corresponding function call instruction, just return None."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ] 
        response = chat(
            model=self._model_path,
            messages=messages,
        )
        return response

    def func_response(self, text, func_map):
        response = self.get_chat(text)
        print("response:", response)
        content = response['message']['content']
        print("content:", content)
    
        try:
            content = json.loads(content)
            func_name = content.get('function', '').lower()
            if not func_name:
                print("No function name")
                return False
            if func_name not in func_map:
                print(f"function name {func_name} not in function map")
                return False
            args = content.get('arguments', {})
            print("start to execute function:", func_name)
            if not args:
                func_map[func_name]()
                return True
            else:
                func_map[func_name](**args)
                return True
        except Exception as e:
            return False

    def get_function_name(self, text, func_map):
        response = self.get_chat(text)
        print("response:", response)
        content = response['message']['content']
        print("content:", content)
    
        try:
            content = json.loads(content)
            func_name = content.get('function', '').lower()
            if not func_name:
                print("No function name")
                return False, None, None
            if func_name not in func_map:
                print(f"function name {func_name} not in function map")
                return False, None, None
            args = content.get('arguments', {})
            return True, func_name, args or None
        except Exception as e:
            return False, None, None
