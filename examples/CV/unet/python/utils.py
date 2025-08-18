import onnxruntime 
import spacemit_ort
import numpy as np
import cv2

def preprocess(img):

    mean_value = [123.675, 116.28, 103.53]
    std_value = [58.395, 57.12, 57.375]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)
    img = (img - np.array(mean_value).astype(np.float32)) / (
        np.array(std_value).astype(np.float32)
    )
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def postprocess(outputs):
    res = outputs[0] / outputs[0].max()
    res = (res * 255.0).astype(np.uint8)
    res = res.reshape([512, 512])
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    return res

def inference(model, img):

    # Create InferenceSession
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 4

    # Load model
    session = onnxruntime.InferenceSession(model,sess_options=session_options, providers=["SpaceMITExecutionProvider"])

    # Get input and output names
    input_name = session.get_inputs()[0].name
    input_dict = {input_name: img}
    output_names = [o.name for o in session.get_outputs()]

    # Run inference
    result = session.run(output_names, input_dict)
    
    return result
