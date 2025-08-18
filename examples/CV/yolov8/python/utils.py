import cv2
import numpy as np
import onnxruntime 
import spacemit_ort

# Load labels
with open('../data/label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]



class Yolov8Detection:
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
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold



    def infer(self, image):
        img = image.copy()

        # Preprocess image                
        input_tensor = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        boxes,classes,scores = self.postprocess(outputs)
        
        # Draw boxes on image
        result_image = self.draw_results(boxes,img, classes, scores)

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


        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))   
        image = np.expand_dims(image, axis=0)    
        return image


    # Postprocess 
    def postprocess(self, output_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(output_data)//defualt_branch
            
        for i in range(defualt_branch):
            boxes.append(self.box_process(output_data[pair_per_branch*i]))
            classes_conf.append(output_data[pair_per_branch*i+1])
            scores.append(np.ones_like(output_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))            

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        
        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)


        return boxes, classes, scores

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)


        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.conf_threshold)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores


    def box_process(self,position):        
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([ self.input_shape[0]//grid_h,  self.input_shape[1]//grid_w]).reshape(1,2,1,1)
        
        position = self.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def dfl(self, position):
        # Distribution Focal Loss DFL)

        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y_max = np.max(y, axis=2, keepdims=True)
        y_exp = np.exp(y - y_max)
        y_sum = np.sum(y_exp, axis=2, keepdims=True)
        y = y_exp / y_sum
        acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        y = (y * acc_matrix).sum(axis=2)

        return y

    # Non-maximum suppression
    def nms(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou_threshold)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep    



    # Draw boxes on image
    def draw_results(self, box, src_img, classes, scores):
        bbox = box.copy()
        shape = src_img.shape[:2]
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = self.input_shape[1] - new_unpad[0], self.input_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        bbox[:, 0] = (bbox[:, 0] - dw) / r
        bbox[:, 0] = np.clip(bbox[:, 0], 0, shape[1])
        bbox[:, 1] = (bbox[:, 1] - dh) / r
        bbox[:, 1] = np.clip(bbox[:, 1], 0, shape[0])
        bbox[:, 2] = (bbox[:, 2] - dw) / r
        bbox[:, 2] = np.clip(bbox[:, 2], 0, shape[1])
        bbox[:, 3] = (bbox[:, 3] - dh) / r
        bbox[:, 3] = np.clip(bbox[:, 3], 0, shape[0])

        for i in range(len(bbox)):
            class_label = labels[classes[i]]
            score = scores[i]
            box = bbox[i]
            cv2.rectangle(src_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(src_img, class_label + " " + str(round(score, 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        return src_img
