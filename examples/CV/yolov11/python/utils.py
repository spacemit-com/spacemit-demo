import cv2
import numpy as np
import onnxruntime 
import spacemit_ort

# Load labels
with open('../data/label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]



class Yolov11Detection:
    def __init__(self, model_path, conf_threshold=0.25,iou_threshold=0.45):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

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
        img = image.copy()

        # Preprocess image                
        input_tensor = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        output = outputs[0]
        offset = output.shape[1]
        anchors = output.shape[2]
        # Postprocess
        dets = self.postprocess(image,output, anchors, offset, conf_threshold=self.conf_threshold)
        final_dets = self.nms(dets,self.iou_threshold)

        # Draw boxes on image
        result_image = self.draw_results(image, final_dets,labels)

        return result_image
    
    # Preprocess image
    def preprocess(self, image):
        shape = image.shape[:2]
        pad_color=(0,0,0)
        
        # Scale ratio    
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        # Compute padding
        ratio = r 
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.input_shape[1] - new_unpad[0], self.input_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  
        dh /= 2    
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border


        
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)        
        image = np.transpose(image, (2, 0, 1))   
        image = np.expand_dims(image, axis=0)    
        return image


    # Postprocess 
    def postprocess(self,image,output, anchors, offset, conf_threshold):
        # Get image shape
        shape = image.shape[:2]
        # Compute scale ratio
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        # Compute unpad offset
        dw, dh = self.input_shape[1] - new_unpad[0], self.input_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        output = output.squeeze()

        # Extract anchor info(x,y,w,h)
        center_x = output[0, :anchors]
        center_y = output[1, :anchors]
        box_width = output[2, :anchors]
        box_height = output[3, :anchors]

        # Extract class probabilities
        class_probs = output[4:offset, :anchors]

        # Find the index of the class with the highest probability
        max_prob_indices = np.argmax(class_probs, axis=0)
        max_probs = class_probs[max_prob_indices, np.arange(anchors)]

        # Filter out low-confidence detections
        valid_mask = max_probs > conf_threshold
        valid_center_x = center_x[valid_mask]
        valid_center_y = center_y[valid_mask]
        valid_box_width = box_width[valid_mask]
        valid_box_height = box_height[valid_mask]
        valid_max_prob_indices = max_prob_indices[valid_mask]
        valid_max_probs = max_probs[valid_mask]

        # Compute bounding box coordinates
        half_width = valid_box_width / 2
        half_height = valid_box_height / 2
        x1 = np.maximum(0, ((valid_center_x - half_width) - dw) / r).astype(int)
        x2 = np.maximum(0, ((valid_center_x + half_width) - dw) / r).astype(int)
        y1 = np.maximum(0, ((valid_center_y - half_height) - dh) / r).astype(int)
        y2 = np.maximum(0, ((valid_center_y + half_height) - dh) / r).astype(int)

        # Combine results
        objects = np.column_stack((x1, y1, x2, y2, valid_max_prob_indices, valid_max_probs)).tolist()
        
        return objects


    # Non-maximum suppression
    def nms(self, dets, iou_threshold=0.45):
        if len(dets) == 0:
            return np.empty((0, 6))
        
        dets_array = np.array(dets)
        # Divide detections into classes
        unique_labels = np.unique(dets_array[:, 4])
        final_dets = []

        for label in unique_labels:
            # Get detections of the current class        
            mask = dets_array[:, 4] == label
            dets_class = dets_array[mask]

            # Sort detections by score in descending order
            order = np.argsort(-dets_class[:, 5])
            dets_class = dets_class[order]

            # Perform non-maximum suppression
            keep = []
            while dets_class.shape[0] > 0:
                
                keep.append(dets_class[0])
                if dets_class.shape[0] == 1:
                    break
                ious = self.calculate_iou(keep[-1], dets_class[1:])            
                dets_class = dets_class[1:][ious < iou_threshold]
            
            final_dets.extend(keep)

        return final_dets

    # Calculate IoU between two boxes
    def calculate_iou(self,box, boxes):
        """
        计算一个框与一组框的 IoU
        :param box: 单个框 [x1, y1, x2, y2]
        :param boxes: 一组框 [N, 4]
        :return: IoU 值 [N]
        """
        # Compute intersection areas
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Compute union areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        # Return intersection over union
        return inter_area / union_area



    # Draw boxes on image
    def draw_results(self, image, dets, class_names):
        image = image.copy()

        for det in dets:
            x1, y1, x2, y2, label, score = det
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{class_names[int(label)]}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image
