import numpy as np
import onnxruntime as ort
from typing import Tuple, List
import cv2

class CLIPSeg:
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    STD = np.array([0.26862954, 0.26130258, 0.27577711])

    def __init__(self, model_path: str,) -> None:
        so = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, so, providers=providers)
        self._build_tokenizer()

    def _build_tokenizer(self):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("../clip-vit-base-patch32")

    # ---------------------------------------------------------------------
    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (352, 352), interpolation=cv2.INTER_LINEAR)
        rgb = (rgb - CLIPSeg.MEAN) / CLIPSeg.STD
        return rgb.transpose(2, 0, 1)[None]

    def _tokenize(self, text: str):
        t = self.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="np")
        return t["input_ids"].astype(np.int64), t["attention_mask"].astype(np.int64)

    # ---------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray, prompt: str) -> np.ndarray:
        pix = self._preprocess(img_bgr).astype(np.float32)
        ids, attn = self._tokenize(prompt)                
        logits = self.session.run(None, {
            "pixel_values": pix,
            "input_ids": ids,
            "attention_mask": attn,
        })[0]
        
        if logits.ndim == 4:
            logits = logits[0, 0]
        elif logits.ndim == 3:
            logits = logits[0]
        elif logits.ndim != 2:
            raise RuntimeError(f"Unexpected CLIPSeg output shape {logits.shape}")
        prob = 1 / (1 + np.exp(-logits.astype(np.float32)))  # sigmoid
        return prob  # (352,352)


def largest_component(mask_bin: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    num, labels, stats, cent = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin, [], []
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    sel = (labels == idx).astype(np.uint8)
    x, y, w, h, _ = stats[idx]
    cx, cy = cent[idx]
    return sel, [x, y, x + w, y + h], [int(cx), int(cy)]
