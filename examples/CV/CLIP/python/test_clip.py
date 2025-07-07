#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from utils import CLIPSeg, largest_component




def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--img", required=True, help="Input image path")
    # text mode
    p.add_argument("--clipseg-model", default="../model/model.onnx", help="CLIPSeg ONNX model path")
    p.add_argument("--text", required=True, help="Text prompt for target")
    p.add_argument("--thr", type=float, default=0.28, help="Probability threshold for CLIPSeg")
    p.add_argument("--kernel", type=int, default=15, help="Dilation kernel size before CCA")
    return p.parse_args()



def main():
    args = parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)


    
    clipseg = CLIPSeg(args.clipseg_model)
    
    prob = clipseg(img, args.text)                 # (352,352)
    prob = cv2.resize(prob, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)

    mask_bin = (prob > args.thr).astype(np.uint8)
    if args.kernel > 1:
        kernel = np.ones((args.kernel, args.kernel), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

    sel, box, point = largest_component(mask_bin)
    if not box:
        raise RuntimeError("CLIPSeg failed to find a region – try lowering --thr or using a better model")
    
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imwrite("result.jpg", img)
    print("save to result.jpg")

if __name__ == "__main__":
    main()
