import re
from typing import Iterable, List, Tuple
import cn2an
from english_utils.abbreviations import expand_abbreviations
from english_utils.time_norm import expand_time_english
from english_utils.number_norm import normalize_numbers as replace_numbers_en


def merge_short_sentences_zh(sens):
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


def split_sentences_zh(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub('[\n\t ]+', ' ', text)
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', text)]
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def replace_numbers_zh(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def replace_punctuation(text):
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "“": "'",
        "”": "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }

    for k, v in rep_map.items():
        text = text.replace(k, v)
    return text


class Lexicon:
    def __init__(self, lexion_filename: str, tokens_filename: str):
        tokens = dict()
        with open(tokens_filename, encoding="utf-8") as f:
            for line in f:
                s, i = line.split()
                tokens[s] = int(i)

        lexicon = dict()
        with open(lexion_filename, encoding="utf-8") as f:
            for line in f:
                splits = line.split()
                word_or_phrase = splits[0]
                phone_tone_list = splits[1:]
                assert len(phone_tone_list) & 1 == 0, len(phone_tone_list)
                phone_str = phone_tone_list[: len(phone_tone_list) // 2]
                phones = [tokens[p] for p in phone_str]

                tones = phone_tone_list[len(phone_tone_list) // 2 :]
                tones = [int(t) for t in tones]

                lexicon[word_or_phrase] = (phone_str, phones, tones)
        lexicon["呣"] = lexicon["母"]
        lexicon["嗯"] = lexicon["恩"]
        self.lexicon = lexicon

        punctuation = ["!", "?", "…", ",", ".", "'", "-"]
        for p in punctuation:
            i = tokens[p]
            tone = 0
            self.lexicon[p] = ([p], [i], [tone])
        self.lexicon[" "] = ([" "], [tokens["_"]], [0])

    def g2p_zh_mix_en(self, text: str) -> Tuple[List[int], List[int]]:
        phone_str = []
        phones = []
        tones = []

        if text not in self.lexicon:
            # print("t", text)
            if len(text) > 1:
                for w in text:
                    # print("w: ", w)
                    s, _, p, t = self.convert(w)
                    if p:
                        phone_str += s
                        phones += p
                        tones += t
            return phone_str, phones, tones

        phone_str, phones, tones = self.lexicon[text]
        return phone_str, phones, tones
    
    
    def split_zh_en(self, text):
        if re.search(r'[a-zA-Z]+', text):
            spliter = '#$&^!@'
            # replace all english words
            text = re.sub(r'[a-zA-Z]+', lambda x: f'{spliter}{x.group()}{spliter}', text)
            texts = text.split(spliter)
            texts = [t for t in texts if len(t) > 0]
            return texts
        else:
            return [text]
        
    
    def normalize_english(self, text):
        text = text.lower()
        text = expand_time_english(text)
        text = replace_numbers_en(text)
        text = expand_abbreviations(text)
        return text

    def normalize_chinese(self, text):
        text = replace_numbers_zh(text)
        return text
    

    def is_english(self, text):
        return 1 if re.match(r'[a-zA-Z\s]+', text) else 0

    def convert(self, text: Iterable[str]) -> Tuple[List[int], List[int]]:
        phone_str = []
        yinjie_num = []
        phones = []
        tones = []

        text = replace_punctuation(text)
        texts_zh_en = self.split_zh_en(text)
        en_num = sum([self.is_english(i) for i in texts_zh_en])
        if en_num * 2 >= len(texts_zh_en):
            texts_zh_en = self.split_zh_en(self.normalize_english(text))
        else:
            texts_zh_en = self.split_zh_en(self.normalize_chinese(text))
        for text_one_lang in texts_zh_en:
            if self.is_english(text_one_lang):
                # English
                s, p, t = self.g2p_zh_mix_en(text_one_lang)

                phone_str += s
                yinjie_num.append(len(s))
                phones += p
                tones += t
            else:
                # print(f"text_one_lang = {text_one_lang}")
                for tl in text_one_lang:
                    s, p, t = self.g2p_zh_mix_en(tl)

                    phone_str += s
                    yinjie_num.append(len(s))
                    phones += p
                    tones += t
            
        return phone_str, yinjie_num, phones, tones