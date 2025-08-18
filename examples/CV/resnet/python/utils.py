import onnxruntime
import spacemit_ort
import cv2
import numpy as np
from PIL import Image

def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    
    return img

def get_image(path):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    return img


def inference(model, img):
    
    # Create InferenceSession
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 4

    # Load model
    session = onnxruntime.InferenceSession(model,sess_options=session_options, providers=["SpaceMITExecutionProvider"])

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    result = session.run([output_name], {input_name: img})[0]    
    
    return result