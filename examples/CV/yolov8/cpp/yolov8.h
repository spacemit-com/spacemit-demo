#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"


struct Object {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float score;
};

const int dfl_len = 16;
const std::string labelFilePath = "../../data/label.txt";
const float conf_threshold = 0.25;
const float iou_threshold = 0.45;
const int classNum = 80;


void Get_Dets(const cv::Mat& image, const float* boxes, const float* scores, const float* score_sum, std::vector<int64_t> dims, int tensor_width, int tensor_height, std::vector<Object>& objects);
Object Dfl(const float* boxes, int anchor_idx, int anchors,  int grid_w, float scale_w, float scale_h, float scale2orign, int pad_w, int pad_h);
cv::Mat Preprocess(const cv::Mat& image, int inputWidth, int inputHeight);
void DrawResults(cv::Mat& image, const std::vector<Object>& dets, const std::vector<std::string>& labels);
float Calculate_Iou(const Object& det1, const Object& det2);
std::vector<Object> Nms(const std::vector<Object>& dets);
std::vector<std::string> ReadLabels();
std::vector<Object> Postprocess(cv::Mat &image, std::vector<Ort::Value>& outputs, size_t output_num, const int inputWidth, const int inputHeight, std::vector<Object> &objects);
cv::Mat Yolov8Inference(cv::Mat& image, const std::string& modelPath);