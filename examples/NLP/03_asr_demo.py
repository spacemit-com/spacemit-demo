import os
import time
import numpy as np

from spacemit_asr import ASRModel
from spacemit_audio import RecAudioVadThread
from spacemit_audio.record_vad import find_audio_device

asr_model = ASRModel()
device_index = find_audio_device()
rec_audio = RecAudioVadThread(sld=1, max_time=5, channels=1, rate=48000, device_index=3, trig_on=0.30, trig_off=0.35)

if __name__ == '__main__':
    try:        
        while True:
            print("Press enter to start!")
            input() # enter 触发

            # 开始录制用户声音
#            rec_audio.max_time_record = 5
#            rec_audio.start_recording()

#            rec_audio.stop_recording() # 等待录音完成
            audio_ret = rec_audio.record_audio() # 获取录音文件路径

            if audio_ret is not None:
                text = asr_model.generate(audio_ret)  # 默认16kHz
                print('user: ', text)
            else:
                print('未获取到有效音频数据')

    except KeyboardInterrupt:
        print("process was interrupted by user.")
    finally:
        rec_audio.cleanup()
