#include "yolov8-seg.h"



int main(int argc, char **argv) {
    
    std::string model_path;
    std::string image_path;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model") && (i < (argc - 1))) {
            model_path = argv[++i];
        } else if ((arg == "--image") && (i < (argc - 1))) {
            image_path = argv[++i];
        }
    }

    std::string default_image_path = "../../data/test.jpg";
    if (image_path.empty()) {
        image_path = default_image_path;
        std::cout << "No image found. Use the default image: " << default_image_path << std::endl;
    }

    std::string default_model_path = "../../model/yolov8-seg_320.q.onnx";
    if (model_path.empty()) {
        model_path = default_model_path;
        std::cout << "No model found. Use the default model: " << default_model_path << std::endl;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image." << std::endl;
        return -1;
    }

    cv::Mat result_image = Yolov8SegInference(image, model_path);

    cv::imwrite("result.jpg", result_image);
    printf("Result image saved as result.jpg\n");

    return 0;


}