import cv2
import numpy as np
import onnxruntime 
import spacemit_ort

# Load labels
with open('../data/label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]


class Yolov5Detection:
    def __init__(self, model_path, conf_threshold=0.4):

        self.conf_threshold = conf_threshold        

        # Create inference session
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 4

        # Load onnx model
        self.session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=["SpaceMITExecutionProvider"])

        # Get input and output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape[2:4]



    def infer(self, image):    
        image_shape = image.shape[:2]

        # Preprocess image                
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        dets = outputs[0].squeeze()
        labels_pred = outputs[1].squeeze()    
        # Postprocess detections        
        scale_x = image_shape[1] / self.input_shape[0]
        scale_y = image_shape[0] / self.input_shape[1]
        dets[:, 0] *= scale_x
        dets[:, 1] *= scale_y
        dets[:, 2] *= scale_x
        dets[:, 3] *= scale_y
        # Draw boxes on image
        result_image = self.draw_results(image, dets, labels_pred, labels)

        return result_image
        

    # Preprocess image
    def preprocess(self, image):        
        image = cv2.resize(image, self.input_shape)        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image


    # Draw boxes on image
    def draw_results(self,image, dets, labels_pred, labels):    
        dets = dets.squeeze()
        labels_pred = labels_pred.squeeze()    

        for i in range(len(dets)):
            det = dets[i]
            score = det[4]
            if score > self.conf_threshold:
                class_id = int(labels_pred[i])        
                x1, y1, x2, y2 = map(int, det[:4])        
                label = labels[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

        

