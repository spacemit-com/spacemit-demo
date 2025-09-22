#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"



#ifdef USE_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "Remap.h"
#include "utils.h"
#endif


struct Letterbox_t {
    int scaled_width ;
    int scaled_height ;
    int offset_width ;
    int offset_height ;
    float scale_ratio ;
};


struct KeyPoint {
    int x;
    int y;
    float visibility;
};

struct Object {
    int x1;
    int y1;
    int x2;
    int y2;    
    float score;
    std::vector<float> source_keypoints;
    std::vector<KeyPoint> keypoints;  // 17个关键点

};


const float conf_threshold = 0.25;
const float iou_threshold = 0.45;
const float point_confidence_threshold = 0.2;



cv::Mat Preprocess(const cv::Mat& image, int inputWidth, int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<Object>& dets, const std::vector<std::string>& labels);
float Calculate_Iou(const Object& det1, const Object& det2);
std::vector<Object> Nms(const std::vector<Object>& dets, float iou_threshold);
std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, int des_width, int des_height);
cv::Mat Yolov8PoseInference(cv::Mat& image, const std::string& modelPath);
Letterbox_t ComputeLetterbox(const cv::Mat& image, int dst_width, int dst_height);
void GetMapXY(const cv::Mat& src, cv::Mat& map_x, cv::Mat& map_y, Letterbox_t letterbox);