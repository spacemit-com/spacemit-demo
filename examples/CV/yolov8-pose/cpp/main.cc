#include <iostream>
#include "yolov8-pose.h"



int main(int argc, char** argv)
{
    std::string modelPath;
    std::string imagePath;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        }
    }
 
    if (modelPath.empty() || imagePath.empty()) {
        std::cout << "Usage: " << argv[0] << " --model <path_to_model> --image <path_to_image>" << std::endl;
        return -1;
    }

    // Load image
    cv::Mat src_image = cv::imread(imagePath);
    if (src_image.empty()) {
        std::cerr << "Error: Could not read image." << std::endl;
        return -1;
    }
    

    cv::Mat image = src_image.clone();
    // Inference
    cv::Mat result_img = Yolov8PoseInference(image, modelPath);


    cv::imwrite("result.jpg", result_img);
    printf("Result image saved as result.jpg\n");

    return 0;
}