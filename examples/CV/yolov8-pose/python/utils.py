import cv2
import numpy as np
import onnxruntime 
import spacemit_ort



class Yolov8Pose:
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
        
        
        # Postprocess
        dets = self.postprocess(image,outputs, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
        
        # final_dets = self.nms(dets,self.iou_threshold)
        
         # Draw boxes on image
        result_image = self.draw_results(image, dets)

        return result_image
    
    # Preprocess image
    def preprocess(self, image):
        shape = image.shape[:2]
        pad_color=(0,0,0)
        
        # Scale ratio    
        r = min(self.input_shape[0] / shape[0], self.input_shape[1] / shape[1])
        # Compute padding        
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
    def postprocess(self,image,output, conf_threshold,iou_threshold):
        
        output = output[0].squeeze()  #   # 去掉 batch 维度
        
        H, W = image.shape[:2]
        input_H, input_W = self.input_shape
    
        # 计算缩放比例和填充量
        r = min(input_H / H, input_W / W)
        dw = (input_W - W * r) / 2
        dh = (input_H - H * r) / 2
        
        # 提取所有检测框和关键点（向量化操作）
        num_detections = output.shape[1]
        if num_detections == 0:
            return []
        
        # 提取检测框坐标 (cx, cy, w, h)
        cx = output[0, :]
        cy = output[1, :]
        w = output[2, :]
        h = output[3, :]
        
        # 转换为(x1, y1, x2, y2)并限制在输入图像范围内
        x1 = np.maximum(0, (cx - w / 2).astype(np.int32))
        y1 = np.maximum(0, (cy - h / 2).astype(np.int32))
        x2 = np.minimum(input_W, (cx + w / 2).astype(np.int32))
        y2 = np.minimum(input_H, (cy + h / 2).astype(np.int32))
        
        # 提取置信度并过滤低置信度检测
        scores = output[4, :]
        mask = scores >= conf_threshold
        if np.sum(mask) == 0:
            return []
        
        # 应用掩码提取有效检测
        valid_indices = np.where(mask)[0]
        x1 = x1[valid_indices]
        y1 = y1[valid_indices]
        x2 = x2[valid_indices]
        y2 = y2[valid_indices]
        scores = scores[valid_indices]
        
        # 提取关键点 (17 * 3: x, y, visibility)
        keypoints = output[5:, valid_indices].T  # 形状: [num_valid, 51]
        keypoints = keypoints.reshape(-1, 17, 3)  # 形状: [num_valid, 17, 3]
        
        # 应用高效NMS
        boxes = np.column_stack((x1, y1, x2, y2))
        keep_indices = self.nms(boxes, scores, iou_threshold)
        if len(keep_indices) == 0:
            return []
        
        # 调整坐标到原始图像尺寸
        x1 = ((x1[keep_indices] - dw) / r).astype(np.int32)
        y1 = ((y1[keep_indices] - dh) / r).astype(np.int32)
        x2 = ((x2[keep_indices] - dw) / r).astype(np.int32)
        y2 = ((y2[keep_indices] - dh) / r).astype(np.int32)
        
        # 调整关键点坐标
        adjusted_keypoints = []
        for kp in keypoints[keep_indices]:
            adjusted_kp = []
            for point in kp:
                x = int((point[0] - dw) / r)
                y = int((point[1] - dh) / r)
                vis = point[2]
                adjusted_kp.append((x, y, vis))
            adjusted_keypoints.append(adjusted_kp)
        
        # 整理结果
        detections = []
        for i in range(len(keep_indices)):
            detections.append({
                'box': [x1[i], y1[i], x2[i], y2[i]],            
                'score': scores[i],
                'keypoints': adjusted_keypoints[i]
            })        
        return detections
        
    # Non-maximum suppression
    def nms(self,boxes, scores, iou_threshold=0.45):
        indices = np.argsort(scores)[::-1]  # 按置信度降序排序
        keep = []
        while indices.size > 0:
            i = indices[0]  # 当前最高置信度框
            keep.append(i)
            # 计算与其余框的IoU
            if indices.size > 1:
                ious = self.calculate_iou(boxes[i], boxes[indices[1:]])
                # 保留IoU < threshold的框
                indices = indices[1:][ious < iou_threshold]
            else:
                break
        return np.array(keep)    
    
    def calculate_iou(self, box, boxes):
        """
        计算单个框与多个框之间的IoU
        Args:
            box: [4] 单个检测框 (x1, y1, x2, y2)
            boxes: [N, 4] 多个检测框
        Returns:
            ious: [N] IoU值
        """
        # 计算交集区域坐标
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        # 计算交集面积
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算当前框和其余框的面积
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 计算并集面积
        union = box_area + boxes_area - intersection
        
        # 计算IoU
        ious = intersection / union
        return ious
    
    # Draw boxes on image
    def draw_results(self, image, detections, box_color=(0, 255, 0), kp_color=(255, 0, 0), line_thickness=2, kp_radius=5, confidence_threshold=0.2):
        KP_CONNECTIONS = [
            [16, 14], [14, 12], [15, 13], [13, 11], [12, 11],  
            [5, 7], [7, 9], [6, 8], [8, 10],  
            [5, 6], [5, 11], [6, 12],  
            [11, 13], [12, 14],  
            [0, 1], [0, 2], [1, 3], [2, 4],  
            [0, 5], [0, 6],  
            [3, 5], [4, 6]  
        ]

        for det in detections:
            # 绘制检测框
            box = det['box']
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, line_thickness)

            # 绘制关键点和连接线
            keypoints = det['keypoints']
            for i, (x, y, vis) in enumerate(keypoints):
                if vis < confidence_threshold:
                    continue  # 跳过不可见关键点
                # 绘制关键点
                cv2.circle(image, (int(x), int(y)), kp_radius, kp_color, -1)

            # 绘制关键点连接线
            for (start_idx, end_idx) in KP_CONNECTIONS:
                if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                    continue  # 防止越界访问
                start_x, start_y, start_vis = keypoints[start_idx]
                end_x, end_y, end_vis = keypoints[end_idx]
                if start_vis < confidence_threshold or end_vis < confidence_threshold:
                    continue
                cv2.line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), kp_color, line_thickness)

        return image