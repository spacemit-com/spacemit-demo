import cv2
import numpy as np
import onnxruntime 
import spacemit_ort
import time


class YoloV8Seg:
    def __init__(self, model_path, conf_threshold=0.25,iou_threshold=0.45):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Create inference session
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1

        # Load onnx model
        self.session = onnxruntime.InferenceSession(model_path,sess_options=session_options, providers=["SpaceMITExecutionProvider"])

        # Get input and output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape[2:4]
        self.src_colors = [(4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183), (17, 31, 104), (255, 111, 221), (255, 68, 79), (204, 237, 0), (0, 243, 68), (189, 0, 255), (0, 180, 255), (221, 0, 186), (0, 255, 255), (38, 192, 0), (1, 255, 179), 
          (125, 36, 255), (123, 0, 104), (255, 27, 108), (252, 109, 47), (162, 255, 11)]
        self.labels  = [line.strip() for line in open('../data/label.txt') if line.strip()]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold


    def infer(self, image):
        img = image.copy()

        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names,{self.input_name : input_tensor})  

        result_image = self.postprocess(img, outputs)
        

        return result_image


         
    # 图像预处理函数
    def preprocess(self, image):
        shape = image.shape[:2]
        pad_color=(0,0,0)
        #调整图像大小
        # Scale ratio
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        # Compute padding    
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        dw, dh = self.input_shape[1] - new_unpad[0], self.input_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border    
        
        # 归一化处理
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)        
        # 调整维度以匹配模型输入 [batch, channel, height, width]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, src_img, input_data):
        boxes, scores, classes_conf, seg_part = [], [], [], []
        defualt_branch = 3
        branch_element = 3
        # Python 忽略 score_sum 输出0        
        output_proto = input_data[-1]

        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[branch_element*i]))
            classes_conf.append(input_data[branch_element*i+1])
            scores.append(np.ones_like(input_data[branch_element*i+1][:,:1,:,:], dtype=np.float32))   
            seg_part.append(input_data[branch_element*defualt_branch + i])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        
        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        seg_part = [sp_flatten(_v) for _v in seg_part]
        
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)
        seg_part = np.concatenate(seg_part)

        # filter according to threshold
        boxes, classes, scores, seg_part = self.filter_boxes(boxes, scores, classes_conf, seg_part)


        # nms
        nboxes, nclasses, nscores, nseg_part = [], [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            sp = seg_part[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])
                nseg_part.append(sp[keep])

        # if not nclasses and not nscores:
        #     return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        seg_part = np.concatenate(nseg_part)
        
        dw, dh, boxes, result_image = self.draw_results(boxes, src_img, classes, scores)
        dw = int(dw)
        dh = int(dh)
        ph, pw = output_proto.shape[-2:]        
        proto = output_proto.reshape(seg_part.shape[-1], -1)
        
        seg_img = np.matmul(seg_part, proto)        
        seg_img = self.sigmoid(seg_img)        
        seg_img = seg_img.reshape(-1,ph, pw)  #seg_img.reshape(-1, ph, pw)
        seg_threadhold = 0.5

        alpha = 0.5
        color_mask = np.zeros((self.input_shape[0]-dh*2,self.input_shape[1]-dw*2,3)).astype(np.uint8)        

        color_size = len(self.src_colors)  
        for n in range (seg_img.shape[0]):    
            image1 = cv2.resize(seg_img[n], self.input_shape, interpolation=cv2.INTER_LINEAR)    
            color_index = int(classes[n]) % color_size
            color = self.src_colors[color_index]
            x1 = int(boxes[n][0])
            y1 = int(boxes[n][1])
            x2 = int(boxes[n][2])
            y2 = int(boxes[n][3])
            image1=image1[dh:self.input_shape[0],dw:self.input_shape[1]]
            roi = image1[y1:y2,x1:x2]       
            mask = roi > seg_threadhold            
            color_mask[y1:y2, x1:x2][mask] = color
            
            overlay = result_image.copy()    
            overlay = cv2.addWeighted(overlay, 1, color_mask, alpha, 0)
                


        return overlay

    def nms_boxes(self,boxes, scores):
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


    def box_process(self,position):           
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_shape[0]//grid_h, self.input_shape[1]//grid_w]).reshape(1,2,1,1)
        
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


    def filter_boxes(self, boxes, box_confidences, box_class_probs, seg_part):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.conf_threshold)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]        
        seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]


        return boxes, classes, scores, seg_part


    def get_real_box(self,box, src_img):
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

        return bbox


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    # 可视化结果
    def visualize_results(self, image, dets, output, output_proto, dw, dh):
        image = image.copy()  
        nseg_part = []
        output = output.squeeze()
            
        unpad_shape = image.shape[:2]  
        alpha=0.5
        color_mask = np.zeros_like(image)
        
        for det in dets:
            x1, y1, x2, y2, label, score, index= det
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            nseg_part.append(output[-32:,int(index)])
            #print(output[84:,int(index)].shape)
            #cv2.putText(image, f'{class_names[int(label)]}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        seg_part = np.array(nseg_part)        
        ph, pw = output_proto.shape[-2:]        
        proto = output_proto.reshape(seg_part.shape[-1], -1)
        
        seg_img = np.matmul(seg_part, proto)        
        seg_img = self.sigmoid(seg_img)        
        seg_img = seg_img.reshape(-1,ph, pw)  #seg_img.reshape(-1, ph, pw)
        seg_threadhold = 0.5
        
        
        color_size = len(self.src_colors)  
        for n in range (seg_img.shape[0]):    
            image1 = cv2.resize(seg_img[n], self.input_shape, interpolation=cv2.INTER_LINEAR)    
            color_index = int(dets[n][4]) % color_size
            color = self.src_colors[color_index]
            x1 = int(dets[n][0])
            y1 = int(dets[n][1])
            x2 = int(dets[n][2])
            y2 = int(dets[n][3])
            image1=image1[dh:self.input_shape[0],dw:self.input_shape[1]]
            roi = image1[y1:y2,x1:x2]       
            mask = roi > seg_threadhold
            color_mask[y1:y2, x1:x2][mask] = color
            
            overlay = image.copy()    
            overlay = cv2.addWeighted(overlay, 1, color_mask, alpha, 0)
                
        return overlay   

    def draw_results(self, box, src_img, classes, scores):
        image = src_img.copy()
        bbox = box.copy()
        shape = src_img.shape[:2]
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = self.input_shape[1] - new_unpad[0], self.input_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        bbox[:, 0] = (bbox[:, 0] - dw) 
        bbox[:, 0] = np.clip(bbox[:, 0], 0, shape[1])
        bbox[:, 1] = (bbox[:, 1] - dh) 
        bbox[:, 1] = np.clip(bbox[:, 1], 0, shape[0])
        bbox[:, 2] = (bbox[:, 2] - dw) 
        bbox[:, 2] = np.clip(bbox[:, 2], 0, shape[1])
        bbox[:, 3] = (bbox[:, 3] - dh) 
        bbox[:, 3] = np.clip(bbox[:, 3], 0, shape[0])        
        image  = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        for i in range(len(bbox)):
            class_label = self.labels[classes[i]]
            score = scores[i]
            box = bbox[i]
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(image, class_label + " " + str(round(score, 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return dw, dh, bbox, image
