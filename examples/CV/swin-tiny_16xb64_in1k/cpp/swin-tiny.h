#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"
#include <fstream>
#include <string>
#include <iostream> 



const std::vector<float> mean = {0.485, 0.456, 0.406};
const std::vector<float> std_dev = {0.229, 0.224, 0.225};


std::vector<std::string> read_imagenet_labels(const std::string& label_file_path);
std::vector<float> preprocess(const cv::Mat& img, int resize_width, int resize_height, int crop_width, int crop_height);
std::string  SwintinyInference(const std::string& model_path, cv::Mat& image);