import os
import time

from functions import func_map

from spacemit_llm import LLMModel, FCModel
llm_model = LLMModel()
fc_model = FCModel()

from spacemit_asr import ASRModel
from spacemit_audio import RecAudioVadThread
asr_model = ASRModel()
rec_audio = RecAudioVadThread(sld=1, max_time=5, channels=1, rate=48000, device_index=3, trig_on=0.60, trig_off=0.35)

if __name__ == '__main__':
    try:
        while True:
            print("Press enter to start!")
            input() # enter 触发
            rec_audio.start_recording()

            rec_audio.stop_recording() # 阻塞录音线程
            audio_ret = rec_audio.get_audio() 
 
            text = asr_model.generate(audio_ret)
            print('user: ', text)

            t1 = time.time()
            function_called = fc_model.func_response(text, func_map)
            print('used time:', time.time() - t1)
            if function_called:
                continue

            llm_output = llm_model.generate(text)
            for output_text in llm_output:
                print(output_text, end='', flush=True)

    except KeyboardInterrupt:
        print("process was interrupted by user.")

