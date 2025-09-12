#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

struct Object {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float score;
    std::vector<std::vector<float>> detect_masks;
};

const int dfl_len = 16;
const std::string labelFilePath = "../../data/label.txt";
const float conf_threshold = 0.25;
const float iou_threshold = 0.45;
const int classNum = 80;
const int branch_element = 4;

const std::vector<cv::Scalar> src_colors = {
    cv::Scalar(4, 42, 255), cv::Scalar(11, 219, 235), cv::Scalar(243, 243, 243), cv::Scalar(0, 223, 183), cv::Scalar(17, 31, 104), cv::Scalar(255, 111, 221), cv::Scalar(255, 68, 79), cv::Scalar(204, 237, 0), cv::Scalar(0, 243, 68), cv::Scalar(189, 0, 255), cv::Scalar(0, 180, 255), cv::Scalar(221, 0, 186), cv::Scalar(0, 255, 255), cv::Scalar(38, 192, 0), cv::Scalar(1, 255, 179), 
    cv::Scalar(125, 36, 255), cv::Scalar(123, 0, 104), cv::Scalar(255, 27, 108), cv::Scalar(252, 109, 47), cv::Scalar(162, 255, 11)
};

cv::Mat Yolov8SegInference(const cv::Mat& image, const std::string& model_path);
cv::Mat Postprocess(cv::Mat &image, std::vector<Ort::Value>& outputs, size_t output_num, const int inputWidth, const int inputHeight, std::vector<Object> &objects);
void Get_Dets(const cv::Mat& image, const float* boxes, const float* scores, const float* score_sum, std::vector<int64_t> dims, int tensor_width, int tensor_height, std::vector<Object>& objects, const float* seg_part, int pad_w, int pad_h);
std::array<float, 4> Dfl(const float* boxes, int anchor_idx, int anchors );
cv::Mat visualize_results(cv::Mat &scaled_image, const std::vector<Object>& objects, float *output_proto, int dw,int dh, std::vector<int64_t> proto_dims, int des_width, int des_height);
inline float sigmoid(float x);
std::vector<std::string> readLabels(const std::string& labelFilePath);
std::vector<Object> Nms(const std::vector<Object>& dets);
float Calculate_Iou(const Object& det1, const Object& det2);
cv::Mat preprocess(const cv::Mat& image, int inputWidth = 640, int inputHeight = 640);