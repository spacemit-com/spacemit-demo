#!/usr/bin/env python3
# vision_chat.py
import argparse
import base64
from pathlib import Path

from spacemit_llm import VisionModel

# ----------------------------------------------------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="文本 + 图像 多模态推理小工具",
    )
    p.add_argument("--model",  default="smolvlm:256m",
                   help="Vision 模型标签(ollama list)")
    p.add_argument("--image",  required=True,
                   help="要推理的图像文件")
    p.add_argument("--prompt", default="",
                   help="首次提问内容；留空则进入交互循环")
    p.add_argument("--stream", default="True",
                   help="是否流式输出")
    return p.parse_args()


# ----------------------------------------------------------------------
def load_b64(img_path: Path) -> str:
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    return base64.b64encode(img_path.read_bytes()).decode()


# ----------------------------------------------------------------------
def main() -> None:
    args = build_args()

    # 1) 载入模型
    llm = VisionModel(vision_model_path=args.model, stream=args.stream)

    # 2) 读图转 base64，一张图重复使用即可
    img_b64 = load_b64(Path(args.image))

    # 3) 如果命令行已经给 prompt → 直接跑一次
    if args.prompt:
        run_once(llm, args.prompt, img_b64)
    else:
        # 否则进入 REPL
        try:
            while True:
                prompt = input("请输入内容（Ctrl-C 退出）：")
                run_once(llm, prompt, img_b64)
        except KeyboardInterrupt:
            print("\n已退出")


# ----------------------------------------------------------------------
def run_once(llm, prompt: str, img_b64: str) -> None:
    """调用 VisionModel 并打印结果（支持流式或非流式）"""
    for chunk in llm.generate(prompt, img_b64):
        print(chunk, end="", flush=True)
    print()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()