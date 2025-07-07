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




cv::Mat Preprocess(const cv::Mat& image, int inputWidth , int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<std::vector<float>>& dets, const std::vector<float>& scores, int* labels_pred, const std::vector<std::string>& labels, float conf_threshold = 0.4);
std::vector<std::string> ReadLabels(const std::string& labelFilePath);
cv::Mat Yolov5Inference(cv::Mat& image, const std::string& modelPath, const std::string& labelFilePath, float conf_threshold);