#ifndef YOLOE__HPP
#define YOLOE__HPP
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 
#include <memory>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"
#include <fstream>
#include <algorithm>
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
struct Letterbox_t {
    int scaled_width = 0;
    int scaled_height = 0;
    int offset_width = 0;
    int offset_height = 0;
};
struct Object {
    float x1;
    float y1;
    float x2;
    float y2;
    int class_id;
    float score;
};


class YOLOE {
public:
    YOLOE(const std::string& yolo_worlde_path);
    ~YOLOE();
    void init_yoloe_model(std::vector<std::vector<float>>clipdata,const cv::Mat& image,float conf_threshold, float iou_threshold);
    cv::Mat inference_yoloe_model(cv::Mat& image);
    void Preprocess(const cv::Mat& image, cv::Mat& blob_image);
    std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, float conf_threshold, float iou_threshold, int des_width,int des_height);
    std::vector<Object> Nms(const std::vector<Object>& dets, float iou_threshold = 0.45);
    void DrawResults(cv::Mat& image, const std::vector<Object>& dets, std::vector<std::string>& labels);
    float Calculate_Iou(const Object& det1, const Object& det2);
    std::vector<std::string> ReadLabels(const std::string& labelFilePath);
    std::vector<std::string> labels;
private:
    std::vector<float> text_data;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session1_;
    Ort::AllocatorWithDefaultOptions allocator_;

     // 输入/输出名称
     std::vector<const char*> input1_names_;
     std::vector<const char*> output1_names_;
 
     std::vector<std::string> input1_names;
     std::vector<std::string> output1_names;
     size_t num_inputs;
     size_t num_outputs;

     Letterbox_t letterbox;
     cv::Mat pad_image;
     std::vector<int64_t> input_image_dims;
     std::vector<int64_t> input_text_dims;

     // text_feature_ 数据
    std::vector<float> text_feature_data_;    
    // text_feature_ 形状
    //std::vector<int64_t> text_feature_shape_;    
    float conf_threshold;
    float iou_threshold;
    //std::vector<std::string> labels;
};
#endif
