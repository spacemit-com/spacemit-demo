import cv2
import numpy as np
import argparse


from utils import Recognition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../model/arcface_mobilefacenet_cut.q.onnx', help='path to onnx model')
    parser.add_argument('--image1', type=str, default='../data/face0.png', help='path to test image1')
    parser.add_argument('--image2', type=str, default='../data/face1.png', help='path to test image2')
    args = parser.parse_args()

      

    
    rec = Recognition(args.model)
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    face0_vector = rec.infer(img1)
    face1_vector = rec.infer(img2)
    
    similarity_scores = face0_vector @ face1_vector.T
    print(f"相似度:{(similarity_scores[0][0]*100):.2f}%.")








if __name__ == '__main__':
    main()