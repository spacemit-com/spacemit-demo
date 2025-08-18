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

const std::string labelFilePath = "../../data/label.txt";
const float conf_threshold = 0.25;
const float iou_threshold = 0.45;



cv::Mat Preprocess(const cv::Mat& image, int inputWidth, int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<Object>& dets, const std::vector<std::string>& labels);
float Calculate_Iou(const Object& det1, const Object& det2);
std::vector<Object> Nms(const std::vector<Object>& dets);
std::vector<std::string> ReadLabels();
std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, int des_width,int des_height);
cv::Mat Yolov11Inference(cv::Mat& image, const std::string& modelPath);