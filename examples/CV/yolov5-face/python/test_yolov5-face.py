import cv2
import numpy as np
import argparse

from utils import Detection



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../model/yolov5n-face_320_cut.q.onnx', help='path to onnx model')
    parser.add_argument('--image', type=str, default='../data/test.jpg', help='path to input image')
    parser.add_argument('--use-camera', action='store_true', help='Use camera as input')
    parser.add_argument('--conf-threshold', type=float, default=0.4, help='Confidence threshold')    
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='Confidence threshold')        
    args = parser.parse_args()

            
    det = Detection(args.model,args.conf_threshold,args.iou_threshold)

    
    if args.use_camera:
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        fram_cp = frame.copy()
        if not ret: 
            print("Can't receive frame (stream end?). Exiting ...")
            return
        boxes = det.infer(fram_cp)
        for box in boxes[0]:                        
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        cv2.imshow("frame", frame)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    else:
        img = cv2.imread(args.image)
        boxes = det.infer(img)
        for box in boxes[0]:                        
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        cv2.imwrite("result.jpg", img)
        


















if __name__ == '__main__':
    main()