import cv2 as cv
import numpy as np
import onnxruntime as ort
#import spacemit_ort

import clip


class TextEmbedder:
    def __init__(self, model_name="../model/clip_text.onnx"):        
        self.session = ort.InferenceSession(model_name, providers=["CPUExecutionProvider"])
        self.input_names = [self.session.get_inputs()[i].name for i in range(len(self.session.get_inputs()))]
        self.output_names = [self.session.get_outputs()[i].name for i in range(len(self.session.get_outputs()))]
        

    def __call__(self, text):
        return self.embed_text(text)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text)        

        np_text_token = np.array(text_token, dtype=np.int64)                
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: np_text_token})
        
        txt_feats = np.concatenate(outputs, axis=0)
        norms = np.linalg.norm(txt_feats, axis=1, keepdims=True)
        txt_feats /= norms        
        txt_feats = np.expand_dims(txt_feats, axis=0)

        return txt_feats

class YOLOWORLD:
    def __init__(self, model_path,conf_threshold=0.2,iou_threshold=0.3):        
        
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4        
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.names = []
        self.text_embedder = TextEmbedder()
        self.class_embeddings = None

        self.conf = conf_threshold
        self.iou = iou_threshold        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))                    ]        
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.num_classes = model_inputs[1].shape[1]



    def infer(self, image, class_embeddings):

        input_tensor = self.prepare_input(image,  self.input_height)
        class_embeddings = self.prepare_embeddings(self.class_embeddings)


        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: input_tensor, self.input_names[1]: class_embeddings})

        
        boxes, scores, class_ids  = self.process_output(outputs, self.conf, self.input_height, self.iou)
        result_image = self.draw_results(image, boxes, scores, class_ids)
    
        return result_image
    
    def draw_results(self, image, boxes, scores, class_ids):
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            class_name = self.names[class_id]
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image



    def set_classes(self, classes):
        self.names = classes    
        self.class_embeddings = self.text_embedder(classes)



    def prepare_input(self, image, imgsz):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        input_img = cv.resize(input_img, (imgsz, imgsz))

        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def prepare_embeddings(self, class_embeddings):        
        if class_embeddings.shape[1] != self.num_classes:
            delta = self.num_classes - class_embeddings.shape[1]
            class_embeddings = np.pad(
                class_embeddings,
                pad_width=((0, 0), (0, delta), (0, 0)),
                mode='constant',
                constant_values=0
            )
        return class_embeddings.astype(np.float32)


    def process_output(self, output, conf, imgsz, iou):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores >= conf, :]
        scores = scores[scores > conf]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.extract_boxes(predictions, imgsz)

        detections = [(class_id, x, y, w, h, score)
                      for class_id, (x, y, w, h), score in zip(class_ids, boxes, scores)]

        nms_detections = self.apply_nms(detections, iou)

        boxes = []
        scores = []
        class_ids = []
        for det in nms_detections:
            class_id, x_nms, y_nms, w_nms, h_nms, score = det
            boxes.append([x_nms, y_nms, w_nms, h_nms])
            scores.append(score)
            class_ids.append(class_id)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions, imgsz):
        boxes = predictions[:, :4]

        boxes[:, 0] /= imgsz
        boxes[:, 1] /= imgsz
        boxes[:, 2] /= imgsz
        boxes[:, 3] /= imgsz 

        boxes[:, 0] *= self.img_width
        boxes[:, 1] *= self.img_height
        boxes[:, 2] *= self.img_width
        boxes[:, 3] *= self.img_height

        return boxes

    def apply_nms(self, detections, iou_threshold):
        boxes = []
        for det in detections:
            (cls_id, x, y, w, h, confidence) = det
            boxes.append([x, y, w, h, cls_id, confidence])

        sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
        selected_boxes = []

        while len(sorted_boxes) > 0:
            selected_boxes.append(sorted_boxes[0])
            remaining_boxes = []

            for box in sorted_boxes[1:]:
                x1_a, y1_a, w1_a, h1_a, _, _ = selected_boxes[-1]
                x1_b, y1_b, w1_b, h1_b, _, _ = box

                x2_a = x1_a + w1_a
                y2_a = y1_a + h1_a
                x2_b = x1_b + w1_b
                y2_b = y1_b + h1_b

                intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                union = w1_a * h1_a + w1_b * h1_b - intersection

                if union == 0:
                    iou = 0
                else:
                    iou = intersection / union

                if iou < iou_threshold:
                    remaining_boxes.append(box)

            sorted_boxes = remaining_boxes

        nms_detections = []
        for box in selected_boxes:
            x, y, w, h, cls_id, confidence = box
            nms_detections.append((cls_id, x, y, w, h, confidence))

        return nms_detections

