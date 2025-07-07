#include "fcn.h"



std::vector<float> preprocess(const cv::Mat& img) {
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(512, 512));
    resized_img.convertTo(resized_img, CV_32F);
    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 512; ++h) {
            for (int w = 0; w < 512; ++w) {
                float pixel_value = resized_img.at<cv::Vec3f>(h, w)[c];
                pixel_value = (pixel_value - mean_value[c]) / std_value[c];
                input_tensor_values.push_back(pixel_value);
            }
        }
    }
    return input_tensor_values;
}


cv::Mat postprocess(const std::vector<float>& output) {
    float max_val = *std::max_element(output.begin(), output.end());
    std::vector<uchar> output_uchar;
    for (float val : output) {
        uchar uchar_val = static_cast<uchar>((val / max_val) * 255.0);
        output_uchar.push_back(uchar_val);
    }
    cv::Mat res(512, 512, CV_8U, output_uchar.data());
    cv::Mat res_bgr;
    cv::cvtColor(res, res_bgr, cv::COLOR_GRAY2BGR);
    return res_bgr;
}

cv::Mat  FcnInference(const std::string& model_path, cv::Mat& image) {

    // Initialize ONNX runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fcn");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    SessionOptionsSpaceMITEnvInit(session_options);
    //Load ONNX model
    Ort::Session session_(env, model_path.c_str(), session_options);

    // Get input info
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_node_names_;
    std::vector<std::string> input_names_;    
    size_t num_inputs_;
    num_inputs_ = session_.GetInputCount();
    input_node_names_.resize(num_inputs_);
    input_names_.resize(num_inputs_, "");
    for (size_t i = 0; i < num_inputs_; ++i) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);        
        input_names_[i].append(input_name.get());
        input_node_names_[i] = input_names_[i].c_str();                       
    }
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();    
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    int inputWidth = input_dims[3];
    int inputHeight = input_dims[2];

    // Get output info
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_names_;
    size_t num_outputs_;
    num_outputs_ = session_.GetOutputCount();
    output_node_names_.resize(num_outputs_);
    output_names_.resize(num_outputs_, "");

    for (size_t i = 0; i < num_outputs_; ++i) {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names_[i].append(output_name.get());
        output_node_names_[i] = output_names_[i].c_str();
    }
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    int outputWidth = output_dims[3];
    int outputHeight = output_dims[2];


    
    // Run preprocess
    std::vector<float> inputTensor = preprocess(image);
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, inputHeight, inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, inputTensor.data(), inputTensor.size(), input_shape.data(), input_shape.size());

    // Run inference
    std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, num_inputs_, output_node_names_.data(), output_node_names_.size());
    
    // Get output tensor data
    float* output_data = outputs[0].GetTensorMutableData<float>();
    
    size_t output_size = outputWidth * outputHeight;
    std::vector<float> output(output_data, output_data + output_size);
    // Run postprocess
    cv::Mat output_img = postprocess(output);
    
    return output_img;
 
}