import cv2
import argparse
from utils import Yolov5Detection


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv5 ONNX Inference')
    parser.add_argument('--model', type=str, default='../model/yolov5_n.q.onnx' , help='Path to the YOLOv5 ONNX model')
    parser.add_argument('--image', type=str, default='../data/test.jpg', help='Path to the input image')
    parser.add_argument('--use-camera', action='store_true', help='Use camera as input')
    parser.add_argument('--conf-threshold', type=float, default=0.4, help='Confidence threshold')    
    args = parser.parse_args()

    detector = Yolov5Detection(args.model, args.conf_threshold)

    if args.use_camera:        
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            # Inference
            result_image = detector.infer(frame)
            # Show result 
            cv2.imshow('YOLOv5 Inference', result_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        if args.image is None:
            print("Please provide either an image path or use the --use-camera option.")
            return
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Failed to read image: {args.image}")
            return
        
        # Inference
        result_image = detector.infer(image)

        # Save result image
        cv2.imwrite('result.jpg', result_image)
        print("Result image saved to result.jpg")

if __name__ == "__main__":
    main()