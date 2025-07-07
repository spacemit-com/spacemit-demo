import os
import sys

from spacemit_tts import TTSModel, play_audio, play_wav, play_wav_non_blocking

tts_model = TTSModel()
warm_up = "欢迎使用进迭时空端到端项目"
tts_model.ort_predict(warm_up)
print("All models init successfully!", flush=True)

if __name__ == '__main__':
    try:
        while True:
            text = input("请输入你想转换的文本：")
            output_audio = tts_model.ort_predict(text)
            play_audio(output_audio)
    except KeyboardInterrupt:
        print("Program was interrupt by user!")
