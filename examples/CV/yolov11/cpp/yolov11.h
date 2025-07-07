#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>


struct Object {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float score;
};




cv::Mat Preprocess(const cv::Mat& image, int inputWidth, int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<Object>& dets, const std::vector<std::string>& labels);
float Calculate_Iou(const Object& det1, const Object& det2);
std::vector<Object> Nms(const std::vector<Object>& dets, float iou_threshold);
std::vector<std::string> ReadLabels(const std::string& labelFilePath);
std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, float conf_threshold,int des_width,int des_height);
cv::Mat Yolov11Inference(cv::Mat& image, const std::string& modelPath, const std::string& labelFilePath, float conf_threshold, float iou_threshold);