import os
import cv2
import numpy as np
import argparse
from  utils import preprocess, postprocess,inference


def main():

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Unet ONNX Inference')
    parser.add_argument('--model', type=str, default='../model/fcn_r50.q.onnx', help='Path to the Unet ONNX model')
    parser.add_argument('--image', type=str, default='../data/test.jpg', help='Path to the input image')
    args = parser.parse_args()

 
    # Load image
    img = cv2.imread(args.image)

    # Image preprocess
    img = preprocess(img)
    
    # Inference
    outputs = inference(args.model,img)
    
    # Postprocess
    res = postprocess(outputs)

    # Save result
    print("Save result to result.jpg")
    cv2.imwrite("result.jpg", res)

if __name__ == "__main__":
    main()