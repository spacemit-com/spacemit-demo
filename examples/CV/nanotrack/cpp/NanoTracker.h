#ifndef NANOTRACKER_H
#define NANOTRACKER_H

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

// 定义跟踪结果的结构体
struct TrackResult {
    cv::Rect2f bbox;
    float best_score;
};

class NanoTracker {
public:
    // 构造函数，初始化跟踪器
    NanoTracker(const std::string& onnx_backbone1_path,
                const std::string& onnx_backbone2_path,
                const std::string& onnx_ban_head_path);

    // 析构函数，用于可能的清理工作
    ~NanoTracker();

    // 用第一帧图像和边界框初始化跟踪器
    void init(const cv::Mat& img, const cv::Rect2f& bbox);

    // 在后续帧中跟踪目标
    TrackResult track(const cv::Mat& img);

private:
    // 参数
    int score_size_;
    int cls_out_channels_;
    int point_stride_;
    float track_context_amount_;
    int track_exemplar_size_;
    int track_instance_size_;
    float track_penalty_k_;
    float track_window_influence_;
    float track_lr_;

    // 状态变量
    cv::Point2f center_pos_;
    cv::Size2f size_;
    std::vector<float> channel_average_;
    std::vector<float> window_;  // Hanning 窗（扁平化）
    std::vector<std::vector<float>> points;  // 生成的点
    // z_feature_ 数据
    std::vector<float> z_feature_data_;    
   // z_feature_ 形状
   std::vector<int64_t> z_feature_shape_;    

    // ONNX Runtime 成员
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session1_;  // 骨干网络 1（模板）
    std::unique_ptr<Ort::Session> session2_;  // 骨干网络 2（搜索）
    std::unique_ptr<Ort::Session> session3_;  // BAN 头
    Ort::AllocatorWithDefaultOptions allocator_;

    // 输入/输出名称
    std::vector<const char*> input1_names_;
    std::vector<const char*> output1_names_;
    std::vector<const char*> input2_names_;
    std::vector<const char*> output2_names_;
    std::vector<const char*> input3_names_;
    std::vector<const char*> output3_names_;

    // 辅助分配名称
    std::vector<std::string> input1_names;
    std::vector<std::string> output1_names;
    std::vector<std::string> input2_names;
    std::vector<std::string> output2_names;
    std::vector<std::string> input3_names;
    std::vector<std::string> output3_names;

    size_t num_inputs;
    size_t num_outputs;
    cv::Scalar mean_val;
    cv::Mat padded_image;


    std::vector<float> hanning(int size);
    std::vector<float> outer(const std::vector<float>& a, const std::vector<float>& b);
    cv::Mat get_subwindow(const cv::Mat& im,std::vector<float> pos,int model_sz,int original_sz,const cv::Scalar& avg_chans);    
    std::vector<std::vector<float>> generate_points(int stride, int size);
    std::vector<std::vector<float>> corner2center(const std::vector<std::vector<float>>& corner);
        
    
    Ort::Value  create_input_tensor(const cv::Mat& data, const std::vector<int64_t>& shape);
    
    template<typename T>
    std::pair<T*, std::vector<int64_t>> get_tensor_data(Ort::Value& tensor);
};

#endif // NANOTRACKER_H    