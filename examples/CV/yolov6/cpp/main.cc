#include <iostream>
#include "yolov6.h"



int main(int argc, char** argv)
{
    std::string modelPath;
    std::string imagePath;
    std::string labelFilePath = "../../data/label.txt";
    float conf_threshold = 0.4;
    

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
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not read image." << std::endl;
        return -1;
    }

    // Inference
    cv::Mat result_img = Yolov6Inference(image, modelPath, labelFilePath, conf_threshold);


    cv::imwrite("result.jpg", result_img);
    printf("Result image saved as result.jpg\n");

    return 0;
}