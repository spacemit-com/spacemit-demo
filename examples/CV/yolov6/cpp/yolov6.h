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
const float conf_threshold = 0.4;



cv::Mat Preprocess(const cv::Mat& image, int inputWidth , int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<std::vector<float>>& dets,float* scores, int* labels_pred, const std::vector<std::string>& labels);
std::vector<std::string> ReadLabels();
cv::Mat Yolov6Inference(cv::Mat& image, const std::string& modelPath);