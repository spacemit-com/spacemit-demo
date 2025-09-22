#ifndef MOBILESAM_HPP
#define MOBILESAM_HPP
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
#define IMG_SIZE 448
#define MAX_TEXT_LINE_LENGTH 1024
#define MASK_THRESHOLD 0
#define COLOR (int[]){30, 144, 144}

struct Letterbox_t {
    int scaled_width = 0;
    int scaled_height = 0;
    int offset_width = 0;
    int offset_height = 0;
    float scale=0.0;
};

class Mobile_Sam
{
    public:
    Mobile_Sam(const std::string& encoder_model_path,const std::string& decoder_model_path);
    ~Mobile_Sam();
    int max(int a,int b);
    void init_Mobile_Sam(const char* point_coords_path, const char* point_labels_path,const cv::Mat& img);
    void inference_mobilesam_model(const cv::Mat& img);
    void Preprocess(const cv::Mat& image, cv::Mat& blob_image);
    float* read_coords_from_file(const char* filename, int* line_count);
    int count_lines(FILE* file);
    void post_process(float* iou_predictions, float* low_res_masks, int ori_height, int ori_width);
    int argmax(float* arr, int size);
    void resize_by_opencv_fp(float *input_image, int input_height, int input_width, float *output_image, int target_height, int target_width);
    void slice_mask(float* ori_mask, float* slice_mask, int ori_width, int slice_height, int slice_width);
    void crop_mask(float* ori_mask, int size, uint8_t* res_mask, float threshold);
    void draw_mask( cv::Mat& image, uint8_t* mask);
    int clamp(float val, int min, int max);
    uint8_t* res_mask;

    private:
    // ONNX Runtime 成员
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session1_; 
    std::unique_ptr<Ort::Session> session2_; 
    Ort::AllocatorWithDefaultOptions allocator_;

    // 输入/输出名称
    std::vector<const char*> input1_names_;
    std::vector<const char*> output1_names_;
    std::vector<const char*> input2_names_;
    std::vector<const char*> output2_names_;

    // 辅助分配名称
    std::vector<std::string> input1_names;
    std::vector<std::string> output1_names;
    std::vector<std::string> input2_names;
    std::vector<std::string> output2_names;

    size_t num_inputs;
    size_t num_outputs;
    int coords_nums;
    float* ori_point_coords;
    int labels_nums;
    float* point_labels;
    float* mask_input;
    float has_mask_input;
    std::vector<int64_t> score_shape_;  
    std::vector<int64_t> masks_shape_;  
    Letterbox_t letterbox;
    

    Ort::Value  create_input_tensor(const cv::Mat& data, const std::vector<int64_t>& shape);
    template<typename T>
    std::pair<T*, std::vector<int64_t>> get_tensor_data(Ort::Value& tensor);
};
#endif
