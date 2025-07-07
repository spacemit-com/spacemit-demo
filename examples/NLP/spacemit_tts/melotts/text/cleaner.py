import time
from . import cleaned_text_to_sequence
import copy

# language_module_map = {"ZH": chinese, 
#                        'ZH_MIX_EN': chinese_mix, 
#                        "EN": english, 
#                        "JP": japanese}
                    #    'KR': korean,
                    #     'FR': french, 
                    #     'SP': spanish, 
                    #     'ES': spanish, 
                    #     }


# def clean_text(text, language):
#     language_module = language_module_map[language]
#     norm_text = language_module.text_normalize(text)
#     phones, tones, word2ph = language_module.g2p(norm_text)
#     return norm_text, phones, tones, word2ph

def clean_text(text, language):
    start = time.time()
    if language == "ZH":
        from . import chinese as language_module
    elif language == "ZH_MIX_EN":
        from . import chinese_mix as language_module
    elif language == "EN":
        from . import english as language_module
    elif language == "JP":
        from . import japanese as language_module
    elif language == "KR":
        from . import korean as language_module
    elif language == "FR":
        from . import french as language_module
    elif language == "SP" or language == "ES":
        from . import spanish as language_module
    else:
        assert False, f"Unsupported lanuguage: {language}"
    print(f"Load language module take {1000 * (time.time() - start)}ms")

    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph

# def clean_text_bert(text, language, device=None):
#     language_module = language_module_map[language]
#     norm_text = language_module.text_normalize(text)
#     phones, tones, word2ph = language_module.g2p(norm_text)
    
#     word2ph_bak = copy.deepcopy(word2ph)
#     for i in range(len(word2ph)):
#         word2ph[i] = word2ph[i] * 2
#     word2ph[0] += 1
#     bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
#     return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass