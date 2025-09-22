import cv2
import time
import argparse
from utils import OCRProcessor



def main():
    parser = argparse.ArgumentParser(description='OCR Demo')
    parser.add_argument('--model_det', type=str, default='../model/ppocr3_det_fixed.onnx', help='Path to the OCR model')
    parser.add_argument('--model_rec', type=str, default='../model/ppocr_rec.onnx', help='Path to the OCR model')
    parser.add_argument('--dict_path', type=str, default='../data/rec_word_dict.txt', help='Path to the dictionary')
    parser.add_argument('--image_path', type=str, default="../data/train_img_33.jpg",
                        help='Path to the image')
    parser.add_argument('--use-camera', action='store_true', help='Use camera as input')
    parser.add_argument('--use-cpu', action='store_true', help='Use CPU as input')
    parser.add_argument('--save-warp-img', action='store_true', help='Save warp image')

    
    args = parser.parse_args()
        
    ocr = OCRProcessor(args.model_det, args.model_rec, args.dict_path,use_cpu=args.use_cpu, save_warp_img=args.save_warp_img)
    if not args.use_camera:
        img = cv2.imread(args.image_path)        
        result = ocr(img)
        print(result)
    else:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera.")
                break
            result = ocr(frame)
            print(result)
            cv2.imshow("OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
