#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"
#include <fstream>
#include <string>
#include <iostream> 



const std::vector<float> mean_value = {123.675, 116.28, 103.53};
const std::vector<float> std_value = {58.395, 57.12, 57.375};


std::vector<float> preprocess(const cv::Mat& img);
cv::Mat postprocess(const std::vector<float>& output);
cv::Mat  UnetInference(const std::string& model_path, cv::Mat& image);