# recorder.py
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
from scipy.signal import resample

__all__ = ["AudioRecorder", "AudioRecorderThread"]

# ------------- 基础工具 -------------
def resample_audio(wave: np.ndarray, from_rate: int, to_rate: int = 16000) -> np.ndarray:
    if from_rate == to_rate:
        return wave
    new_len = int(len(wave) * to_rate / from_rate)
    return resample(wave.astype(np.float32), new_len).astype(np.int16)

# ------------- 主录音类 -------------
class AudioRecorder:
    """
    只负责麦克风采集
    ----------------------------------
    用法:
        rec = AudioRecorder(rate=44100, seconds=3)
        wave = rec.record()          # numpy[int16] (N,)
    """
    def __init__(
        self,
        seconds: float = 5.0,            # 最长录音时长
        channels: int = 1,
        rate: int = 16000,
        frame_samples: int = 512,
        device_index: Optional[int] = None,
        target_rate: int = 16000,        # 若与 rate 不同则自动重采样
    ):
        self.seconds       = seconds
        self.channels      = channels
        self.rate          = rate
        self.frame_samples = frame_samples
        self.device_index  = device_index
        self.target_rate   = target_rate

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frame_samples,
            input_device_index=self.device_index,
        )

    # ---------- 阻塞录音 ----------
    def record(self) -> np.ndarray:
        """阻塞式采集 self.seconds 秒音频，返回 numpy[int16]."""
        n_frames = int(self.rate * self.seconds / self.frame_samples)
        frames = []

        for _ in range(n_frames):
            frames.append(self._stream.read(self.frame_samples, exception_on_overflow=False))

        # bytes -> numpy[int16]
        raw = b"".join(frames)
        wave = np.frombuffer(raw, dtype=np.int16)

        # 可选重采样到 target_rate
        wave = resample_audio(wave, self.rate, self.target_rate)
        return wave

    # ---------- 资源释放 ----------
    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

# ------- 线程版（开始 / 停止 / 取）-------
class AudioRecorderThread(AudioRecorder):
    """
    rec = AudioRecorderThread(seconds=10)
    rec.start()          # 非阻塞
    ...
    rec.stop()           # 等待录完
    wav = rec.wave       # numpy[int16] 或 None
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread = None
        self.wave: Optional[np.ndarray] = None

    # 非阻塞启动
    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        self.wave = self.record()

    # 阻塞等待结束
    def stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def close(self):
        super().close()

# ----------------- demo -----------------
if __name__ == "__main__":
    rec = AudioRecorderThread(seconds=4, rate=44100, target_rate=16000)
    print("按 Enter 开始录音 4 秒 …")
    input()
    rec.start()
    rec.stop()

    if rec.wave is not None:
        print("采样点数:", rec.wave.shape[0], "dtype:", rec.wave.dtype)
        Path("recorded.pcm").write_bytes(rec.wave.tobytes())
        print("已保存 raw PCM -> recorded.pcm")
    rec.close()