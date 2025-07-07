import os
import threading
from playsound import playsound
import subprocess
play_device = 'plughw:0,0'

def play_audio(wav_file_path):
    try:
        if os.path.exists(wav_file_path):
            playsound(wav_file_path)
        else:
            print(f"{wav_file_path} does not exist")
    except Exception as e:
        print(f"An error occurred while trying to play the audio file: {e}")

def play_wav(path, device=play_device, volume='120%'):
    number = device.split(":")[1].split(",")[0]
    cmd = f'amixer -c {number} set PCM {volume}'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pr = f'Set playback volume to {volume} return {proc.returncode}'
    print(pr)

    # cmd = f'aplay -D{device} -r 48000 -f S16_LE {path}'
    cmd = f'aplay -D {device} {path}'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pr = f'Play {path} on {device} return {proc.returncode}'
    print(pr)

def play_wav_non_blocking(path, device=play_device, volume='80%'):
    # 设置音量
    number = device.split(":")[1].split(",")[0]
    cmd = f'amixer -c {number} set PCM {volume}'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'Set playback volume to {volume} return {proc.returncode}')

    # 非阻塞播放音频
    cmd = f'aplay -D {device} {path}'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'Playing {path} on {device} (PID={proc.pid})')

    return proc  # 返回进程对象，方便后续管理（比如停止播放）