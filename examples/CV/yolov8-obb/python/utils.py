import cv2
import numpy as np
import onnxruntime 
#import spacemit_ort

# Load labels
with open('../data/label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]



class Yolov8OBBDetection:
    def __init__(self, model_path, conf_threshold=0.25,iou_threshold=0.45):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Create inference session
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 4

        # Load onnx model
        self.session = onnxruntime.InferenceSession(model_path)#,sess_options=session_options, providers=["SpaceMITExecutionProvider"])

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

    def infer_dfl(self, image):
        img = image.copy()

        # Preprocess image                
        input_tensor = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})      
        
        boxes,classes,scores = self.postprocess_dfl(outputs)
        boxes = self.get_real_box(boxes,img)
        
        for i in range(len(boxes)):
            class_label = labels[classes[i]]
            score = scores[i]
            box = boxes[i]
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(img, class_label + " " + str(round(score, 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img
    
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
        
    def dfl(self, position):
        # Distribution Focal Loss DFL)
        import torch

        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = y.softmax(2)
        
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        
        y = (y*acc_metrix).sum(2)
        return y.numpy()


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

    def postprocess_dfl(self,input_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出0
        
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))            

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
            keep = self.nms_boxes(b, s)

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


    # Postprocess for YOLOv8-OBB (Oriented Bounding Box)
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

        # Extract OBB parameters (x,y,w,h,angle)
        center_x = output[0, :anchors]
        center_y = output[1, :anchors]
        box_width = output[2, :anchors]
        box_height = output[3, :anchors]

        # Extract class probabilities (indices 4 to offset-2)
        class_probs = output[4:offset-1, :anchors]

        # Extract rotation angle (last dimension)
        rotation_angle = output[-1, :anchors]

        # Find the index of the class with the highest probability
        max_prob_indices = np.argmax(class_probs, axis=0)
        max_probs = class_probs[max_prob_indices, np.arange(anchors)]

        # Filter out low-confidence detections
        valid_mask = max_probs > conf_threshold
        valid_center_x = center_x[valid_mask]
        valid_center_y = center_y[valid_mask]
        valid_box_width = box_width[valid_mask]
        valid_box_height = box_height[valid_mask]
        valid_rotation_angle = rotation_angle[valid_mask]
        valid_max_prob_indices = max_prob_indices[valid_mask]
        valid_max_probs = max_probs[valid_mask]

        # Convert to original image coordinates
        valid_center_x = (valid_center_x - dw) / r
        valid_center_y = (valid_center_y - dh) / r
        valid_box_width = valid_box_width / r
        valid_box_height = valid_box_height / r
        
        # Normalize angles to [-π/2, π/2] range
        valid_rotation_angle = np.array([self.normalize_angle(angle) for angle in valid_rotation_angle])        
        # Combine results as OBB format: [center_x, center_y, width, height, angle, class_id, confidence]
        objects = np.column_stack((valid_center_x, valid_center_y, valid_box_width, 
                                 valid_box_height, valid_rotation_angle, 
                                 valid_max_prob_indices, valid_max_probs)).tolist()
        
        return objects

    def normalize_angle(self, angle):
        """
        将角度归一化到 [-π/2, π/2] 范围内
        """
        while angle > np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
        return angle

    # Non-maximum suppression for OBB
    def nms(self, dets, iou_threshold=0.45):
        if len(dets) == 0:
            return []
        
        dets_array = np.array(dets)
        # Divide detections into classes
        unique_labels = np.unique(dets_array[:, 5])  # class_id is at index 5 for OBB
        final_dets = []

        for label in unique_labels:
            # Get detections of the current class        
            mask = dets_array[:, 5] == label
            dets_class = dets_array[mask]

            # Sort detections by score in descending order
            order = np.argsort(-dets_class[:, 6])  # confidence is at index 6 for OBB
            dets_class = dets_class[order]

            # Perform non-maximum suppression
            keep = []
            while dets_class.shape[0] > 0:
                # Add the detection with highest score
                keep.append(dets_class[0].tolist())
                
                if dets_class.shape[0] == 1:
                    break
                    
                # Calculate IoU with remaining detections
                ious = self.calculate_iou(dets_class[0], dets_class[1:])
                
                # Keep only detections with IoU below threshold
                keep_mask = ious < iou_threshold
                dets_class = dets_class[1:][keep_mask]
            
            final_dets.extend(keep)

        return final_dets

    def obb_to_polygon(self, obb):
        """
        将旋转边界框转换为多边形顶点
        :param obb: [center_x, center_y, width, height, angle]
        :return: 多边形顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        cx, cy, w, h, angle = obb[:5]
        
        # 计算旋转边界框的四个顶点（相对于中心点）
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 半宽和半高
        half_w = w / 2
        half_h = h / 2
        
        # 四个顶点（相对于中心点）
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # 应用旋转
        rotated_corners = corners @ rotation_matrix.T
        
        # 平移到实际位置
        polygon = rotated_corners + np.array([cx, cy])
        
        return polygon

    def polygon_iou(self, poly1, poly2):
        """
        计算两个多边形的IoU
        """        
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            
            if not p1.is_valid or not p2.is_valid:
                return 0.0
                
            intersection = p1.intersection(p2)
            union = p1.union(p2)
            
            if union.area == 0:
                return 0.0
                
            return intersection.area / union.area
        except:
            # 如果shapely不可用，使用简化的矩形IoU作为近似
            return self.calculate_rect_iou(poly1, poly2)

    def calculate_rect_iou(self, poly1, poly2):
        """
        简化的矩形IoU计算（当shapely不可用时使用）
        """
        # 获取边界框
        x1_min, y1_min = np.min(poly1, axis=0)
        x1_max, y1_max = np.max(poly1, axis=0)
        x2_min, y2_min = np.min(poly2, axis=0)
        x2_max, y2_max = np.max(poly2, axis=0)
        
        # 计算交集
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    # Calculate IoU between two OBB boxes
    def calculate_iou(self, box, boxes):
        """
        计算一个OBB框与一组OBB框的 IoU
        :param box: 单个OBB框 [center_x, center_y, width, height, angle, class_id, confidence]
        :param boxes: 一组OBB框 [N, 7]
        :return: IoU 值 [N]
        """
        if len(boxes) == 0:
            return np.array([])
            
        # 确保box是numpy数组
        if not isinstance(box, np.ndarray):
            box = np.array(box)
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
            
        # 提取OBB参数
        box_obb = box[:5]  # [center_x, center_y, width, height, angle]
        boxes_obb = boxes[:, :5]  # [N, 5]
        
        # 转换为多边形
        box_poly = self.obb_to_polygon(box_obb)
        
        ious = []
        for i in range(len(boxes_obb)):
            box_i_poly = self.obb_to_polygon(boxes_obb[i])
            iou = self.polygon_iou(box_poly, box_i_poly)
            ious.append(iou)
            
        return np.array(ious)



    # Draw OBB boxes on image
    def draw_results(self, image, dets, class_names):
        image = image.copy()

        for det in dets:
            # OBB format: [center_x, center_y, width, height, angle, class_id, confidence]
            center_x, center_y, width, height, angle, label, score = det
            
            # Convert OBB to polygon vertices
            obb = [center_x, center_y, width, height, angle]
            polygon = self.obb_to_polygon(obb)
            
            # Convert to integer coordinates for drawing
            polygon_int = polygon.astype(np.int32)
            
            # Draw the rotated rectangle
            cv2.polylines(image, [polygon_int], True, (0, 255, 0), 2)
            
            # Draw label and confidence
            label_text = f'{class_names[int(label)]}: {score:.2f}'
            text_pos = (int(center_x), int(center_y - 10))
            cv2.putText(image, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image
