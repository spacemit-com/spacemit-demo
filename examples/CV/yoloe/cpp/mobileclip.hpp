#ifndef MOBILECLIP_HPP
#define MOBILECLIP_HPP
#include <string>
#include <vector>
#include "mobileclip_tokenizer.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

class MOBILECLIP {
public:
    MOBILECLIPTokenizer* tokenizer;
    MOBILECLIP(const std::string& text_model_path);
    ~MOBILECLIP();
    //int init_clip_model(const char* text_model_path);
    void inference_clip_model(std::vector<std::string> test_strings,int text_num);
    std::vector<std::vector<float>>clip_data_;

private:
    
    std::vector<float> clip_data; 
    std::vector<int64_t> clip_shape;      
    // ONNX Runtime 成员
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
    
    int sequence_len;
    Ort::Value create_input_tensor(const std::vector<int64_t>& data, const std::vector<int64_t>& shape);
    template<typename T>
    std::pair<T*, std::vector<int64_t>> get_tensor_data(Ort::Value& tensor);
    };

#endif 
