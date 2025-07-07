import argparse
import numpy as np
from utils import get_image,preprocess,inference

# 标签加载
labels = []
with open('../data/label.txt', 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../model/inception_v1.q.onnx', help='model name')
    parser.add_argument('--image', type=str, default='../data/kitten.jpg', help='image path') 
    args = parser.parse_args()

    # Load image
    img = get_image(args.image)
    # Image preprocess
    img = preprocess(img)
    # Inference
    result = inference(args.model, img)
    # Postprocess
    result = np.squeeze(result)
    top_k = result.argsort()[-5:][::-1]
    
    
    print ("Final Result: ")
    print('class=%s' %labels[top_k[0]])




if __name__ == "__main__":
    main()
