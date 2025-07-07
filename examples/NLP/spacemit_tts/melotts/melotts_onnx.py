import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import soundfile
import onnxruntime as ort
import argparse
import subprocess
import time
from .split_utils import split_sentence
from .text import cleaned_text_to_sequence
from .text.cleaner import clean_text
from .symbols import *
import tempfile
import re
import spacemit_ort

model_url = "https://archive.spacemit.com/spacemit-ai/openwebui/tts_models.tar.gz"
cache_dir = os.path.expanduser("~/.cache")
tts_model_dir = os.path.join(cache_dir, "tts_models")
tts_model_path = os.path.join(tts_model_dir, "g-zh_mix_en.bin")
tar_path = os.path.join(cache_dir, "tts_models.tar.gz")

language = "ZH_MIX_EN"

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)

    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32) * 2
    word2ph[0] += 1

    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts

def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    # Average pad part
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

# 计算每个词的发音时长
def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

# 生成有overlap的slice，slice索引是对于zp的
def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices = []
    zp_slices = []
    while pn_end < len(word2pronoun):
        # 前一个slice长度大于2 且 加上现在这个字没有超过dec_len，则往前overlap两个字
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
            
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices

class TTSModel:
    def __init__(self, enc_model='encoder-zh.onnx', dec_model='decoder-zh.onnx'):
        if not os.path.exists(tts_model_path):
            print("TTS模型文件不存在，正在下载模型文件")
            subprocess.run(["wget", "-O", tar_path, model_url], check=True)
            subprocess.run(["tar", "-xvzf", tar_path, "-C", cache_dir], check=True)
            subprocess.run(["rm", "-rf", tar_path], check=True)
            print("Models Download successfully")
        self._enc_model = os.path.join(tts_model_dir, enc_model)
        self._dec_model = os.path.join(tts_model_dir, dec_model)
        self.sample_rate = 48000
        self._symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[language])}

        start = time.time()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4 
        self.sess_enc = ort.InferenceSession(self._enc_model, providers=["CPUExecutionProvider"], sess_options=sess_options)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.sess_dec = ort.InferenceSession(self._dec_model, providers=["CPUExecutionProvider"], sess_options=sess_options)
        self.dec_len = 65536 // 512
        print(f"load models take {1000 * (time.time() - start)}ms")

        # Load static input
        self.g = np.fromfile(
            os.path.join(tts_model_dir, f'g-{language.lower()}.bin'),
            dtype=np.float32
            ).reshape(1, 256, 1)
    # g = np.fromfile(f"../models/g-{language.lower()}.bin", dtype=np.float32).reshape(1, 256, 1)


    def ort_predict(self, text):

        # Final wav
        self.audio_list = []

        self.speed = 0.8

        # Split sentence
        start = time.time()
        sens = split_sentences_into_pieces(text, language, quiet=False)
        print(f"split_sentences_into_pieces take {1000 * (time.time() - start)}ms")

        # Iterate over splitted sentences
        for n, se in enumerate(sens):
            if language in ['EN', 'ZH_MIX_EN']:
                se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
            print(f"\nSentence[{n}]: {se}")
            # Convert sentence to phones and tones
            phones, tones, lang_ids, norm_text, word2ph = get_text_for_tts_infer(se, language, symbol_to_id=self._symbol_to_id)

            start = time.time()
            # Run encoder
            z_p, pronoun_lens, audio_len = self.sess_enc.run(
                None, 
                input_feed={
                    'phone': phones, 'g': self.g,
                    'tone': tones, 'language': lang_ids, 
                    'noise_scale': np.array([0], dtype=np.float32),
                    'length_scale': np.array([1.0 / self.speed], dtype=np.float32),
                    'noise_scale_w': np.array([0], dtype=np.float32),
                    'sdp_ratio': np.array([0], dtype=np.float32)
                }
            )
            print(f"encoder run take {1000 * (time.time() - start):.2f}ms")

            # 计算每个词的发音长度
            word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
            # 生成word2pronoun和zp的切片
            pn_slices, zp_slices = generate_slices(word2pronoun, self.dec_len)

            audio_len = audio_len[0]
            sub_audio_list = []
            for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
                zp_slice = z_p[..., zs]

                # Padding前zp的长度
                sub_dec_len = zp_slice.shape[-1]
                # Padding前输出音频的长度
                sub_audio_len = 512 * sub_dec_len

                # Padding到dec_len
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], self.dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

                start = time.time()
                audio = self.sess_dec.run(None, input_feed={"z_p": zp_slice,
                                    "g": self.g
                                    })[0].flatten()
            
                # 处理overlap
                audio_start = 0
                if len(sub_audio_list) > 0:
                    if pn_slices[i - 1].stop > ps.start:
                        # 去掉第一个字
                        audio_start = 512 * word2pronoun[ps.start]
        
                audio_end = sub_audio_len
                if i < len(pn_slices) - 1:
                    if ps.stop > pn_slices[i + 1].start:
                        # 去掉最后一个字
                        audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]

                audio = audio[audio_start:audio_end]
                print(f"Decode slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
                sub_audio_list.append(audio)
            sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
            self.audio_list.append(sub_audio)
        audio = audio_numpy_concat(self.audio_list, sr=self.sample_rate, speed=self.speed)
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file = temp_wav_file.name
        soundfile.write(audio_file, audio, self.sample_rate)
        print(f"Save to {audio_file}")
        return audio_file
