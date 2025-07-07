#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NanoTracker.h"

// 解析命令行参数的结构体
struct Args {
    std::string video_name = "";
};

// 解析命令行参数
Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--video_name" && i + 1 < argc) {
            args.video_name = argv[++i];
        }
    }
    return args;
}



std::string extractVideoName(const std::string& video_path) {
    if (video_path.empty()) return "webcam";

    size_t last_slash = video_path.find_last_of("/\\");
    std::string base_name = (last_slash != std::string::npos) ? video_path.substr(last_slash + 1) : video_path;

    size_t dot_pos = base_name.find_last_of('.');
    return (dot_pos != std::string::npos) ? base_name.substr(0, dot_pos) : base_name;
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    // 假设 ONNX 模型路径固定
    std::string onnx_backbone1 = "../../model/nanotrack_backbone1.onnx";
    std::string onnx_backbone2 = "../../model/nanotrack_backbone2.q.onnx";
    std::string onnx_ban_head = "../../model/nanotrack_head.q.onnx";

    NanoTracker tracker(onnx_backbone1, onnx_backbone2, onnx_ban_head);
    bool first_frame = true;
    std::string video_name = args.video_name;
  
    
    
    
    cv::VideoCapture cap;
    if (args.video_name.empty()) {
        cap.open(1); // 使用默认摄像头
        if (!cap.isOpened()) {
            std::cerr << "Failed to open webcam!" << std::endl;
            return -1;
        }
    } else {
        cap.open(args.video_name);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video file: " << args.video_name << std::endl;
            return -1;
        }
    }
    cv::Mat frame;

    while (cap.read(frame)) {        
        cv::Mat frame_clone = frame.clone();
        if(first_frame) {
            cv::Rect2f init_rect(244, 161, 74, 70);            
            tracker.init(frame_clone, init_rect);
            first_frame = false;
        } else {            
            TrackResult result = tracker.track(frame_clone);
            cv::rectangle(frame, result.bbox, cv::Scalar(0, 255, 0), 3);
            cv::imshow(video_name, frame);
            int key = cv::waitKey(30); // 调整等待时间
            if (key == 27) { // 按 ESC 键退出
            break;
            }
        }
    }


    return 0;
}
    