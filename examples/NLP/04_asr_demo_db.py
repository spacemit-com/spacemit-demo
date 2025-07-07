import os
import time

from spacemit_asr import ASRModel
from spacemit_audio import RecAudioDBThread

asr_model = ASRModel()
rec_audio = RecAudioDBThread(sld=1, min_db=2000, max_time=5, channels=1, rate=48000, device_index=3)

if __name__ == '__main__':
    try:
        while True:
            print("Press enter to start!")
            input() # enter 触发

#            rec_audio.max_time_record = 5
#            rec_audio.start_recording()
#            rec_audio.stop_recording()
            # 开始录制用户声音
            audio_ret = rec_audio.record_audio() # 获取录音文件路径

            text = asr_model.generate(audio_ret)
            print('user: ', text)

    except KeyboardInterrupt:
        print("process was interrupted by user.")
