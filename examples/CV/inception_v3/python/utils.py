import cv2
import numpy as np
from PIL import Image
import onnxruntime
import spacemit_ort



def preprocess(img):
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)    
    img = cv2.resize(img, (342, 342))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 299) // 2
    x0 = (w - 299) // 2
    img = img[y0 : y0+299, x0 : x0+299, :]
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