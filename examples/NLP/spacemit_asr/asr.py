import wave
import os
import time
import subprocess
import warnings
import numpy as np

from .models.sensevoice_bin import SenseVoiceSmall
from .models.postprocess_utils import rich_transcription_postprocess

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/sensevoice.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
asr_model_dir = os.path.join(cache_dir, "sensevoice")
asr_model_path = os.path.join(asr_model_dir, "model_quant_optimized.onnx")
tar_path = os.path.join(cache_dir, "sensevoice.tar.gz")

class ASRModel:
    def __init__(self):
        if not os.path.exists(asr_model_path):
            print("模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")

        self._model_path = asr_model_dir
        print(f"初始化ASR模型，路径: {self._model_path}")
        try:
            # 尝试更保守的配置以提高兼容性
            self._model = SenseVoiceSmall(
                self._model_path, 
                batch_size=1, 
                quantize=True,
                intra_op_num_threads=1,
            )
            print("ASR模型初始化成功")
        except Exception as e:
            print(f"ASR模型初始化失败: {e}")
            raise

    def generate(self, audio_file, sr=16000):
        if isinstance(audio_file, np.ndarray):
            # 将int16音频数据归一化为float32（-1到1范围）
            if audio_file.dtype == np.int16:
                audio_path = audio_file.astype(np.float32) / 32768.0
            else:
                audio_path = audio_file
            audio_dur = len(audio_file) / sr
        elif isinstance(audio_file, str):
            audio_path = [audio_file]
            audio_dur = wave.open(audio_file).getnframes() / sr
        else:
            warnings.warn(
                f"[ASR] Unsupported type {type(audio_file).__name__}; "
                "expect str or np.ndarray. Skip this turn."
            )
            return None
            audio_dur = len(audio_file) / 16000

        print(f"开始ASR推理，音频长度: {audio_dur:.3f}s")
        t0 = time.perf_counter()
        try:
            asr_res = self._model(audio_path, language='zh', use_itn=True)
            print(f"推理完成，结果类型: {type(asr_res)}")
            if hasattr(asr_res, '__len__'):
                print(f"结果长度: {len(asr_res)}")
            if len(asr_res) > 0 and hasattr(asr_res[0], '__len__'):
                print(f"第一层长度: {len(asr_res[0])}")
                if len(asr_res[0]) > 0:
                    print(f"实际内容: {asr_res[0][0]}")
        except Exception as e:
            print(f"[ASR] 推理错误详情: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
        infer_time = time.perf_counter() - t0
        rtf = infer_time / audio_dur if audio_dur > 0 else float("inf")
        print(f"infer_time: {infer_time:.3f}s, audio_dur: {audio_dur:.3f}s, RTF: {rtf:.2f}")
        # 后处理
        asr_res = asr_res[0][0].tolist()
        text = rich_transcription_postprocess(asr_res[0])
        return text
