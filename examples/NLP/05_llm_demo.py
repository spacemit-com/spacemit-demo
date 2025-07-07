import os
import time

from functions import func_map

from spacemit_llm import LLMModel, FCModel
llm_model = LLMModel(llm_model_path='qwen2.5:0.5b', stream=True)
fc_model = FCModel(fc_model_path='qwen2.5-0.5b-fc')

if __name__ == '__main__':
    try:
        while True:
            text = input("请输入内容：")
            t1 = time.time()
            function_called = fc_model.func_response(text, func_map)
            print('used time:', time.time() - t1)
            if function_called:
                continue

            llm_output = llm_model.generate(text)
            for output_text in llm_output:
                print(output_text, end='', flush=True)
            print()
    except KeyboardInterrupt:
        print("process was interrupted by user.")