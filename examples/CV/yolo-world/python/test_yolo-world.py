import cv2
import numpy as np
import argparse
from utils import  YOLOWORLD


def main():
    parser = argparse.ArgumentParser(description='YOLO-World ONNX Inference')
    parser.add_argument('--model', type=str, default='../model/yolov8s-worldv2.onnx', help='Path to the YOLOv8 ONNX model')
    parser.add_argument('--image', type=str, default='../data/test.jpg', help='Path to the input image')
    parser.add_argument('--use-camera', action='store_true', help='Use camera as input')
    parser.add_argument('--conf-threshold', type=float, default=0.2, help='Confidence threshold')    
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='Confidence threshold')
    parser.add_argument('--classes', nargs='+', type=str, default=['people'],help='Input class names：people car telephone')

    args = parser.parse_args()

    # Create detector 
    detector = YOLOWORLD(args.model,args.conf_threshold,args.iou_threshold)
    class_names = args.classes    
    detector.set_classes(class_names)

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
            result_image = detector.infer(frame, class_names)

            # Show result
            cv2.imshow('YOLO-World Inference', result_image)

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
        result_image = detector.infer(image,class_names)

        # Save result image
        cv2.imwrite('result.jpg', result_image)
        print("Results saved to result.jpg")

if __name__ == "__main__":
    main()