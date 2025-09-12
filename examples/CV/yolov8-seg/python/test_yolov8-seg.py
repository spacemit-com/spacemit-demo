import cv2
import numpy as np
import argparse
from utils import YoloV8Seg

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv8 ONNX Inference')
    parser.add_argument('--model', type=str, default='../model/yolov8n_seg_320.q.onnx', help='Path to the YOLOv8 ONNX model')
    parser.add_argument('--image', type=str, default='../data/test.jpg', help='Path to the input image')
    parser.add_argument('--use-camera', action='store_true', help='Use camera as input')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')    
    args = parser.parse_args()

    detector = YoloV8Seg(args.model, conf_threshold=args.conf_threshold)


    if  args.use_camera:
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
        
            result_image = detector.infer(frame)
            # 显示结果
            cv2.imshow('YOLOv8-Seg Inference', result_image)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

        # 释放摄像头并关闭所有窗口
        cap.release()
        cv2.destroyAllWindows()
    else:
        if args.image is None:
            print("Please provide either an image path or use the --use-camera option.")
            return

        image = cv2.imread(args.image)        
        result_image = detector.infer(image)

        # 显示结果
        cv2.imwrite('result.jpg', result_image)
        print("Results saved to result.jpg")


if __name__ == "__main__":
    main()