#include "inception_v1.h"




std::vector<std::string> read_imagenet_labels(const std::string& label_file_path) {
    std::vector<std::string> labels;
    std::ifstream label_file(label_file_path);

    if (!label_file.is_open()) {
        std::cerr << "无法打开标签文件: " << label_file_path << std::endl;
        return labels;
    }

    std::string line;
    while (std::getline(label_file, line)) {
        labels.push_back(line);
    }

    label_file.close();
    return labels;
}

std::vector<float> preprocess(const cv::Mat& img, int resize_width, int resize_height){
    cv::Mat processed_img;
    img.convertTo(processed_img, CV_32F, 1.0 / 255.0);
    
    cv::resize(processed_img, processed_img, cv::Size(resize_width, resize_height));



    for (int y = 0; y < processed_img.rows; ++y) {
        for (int x = 0; x < processed_img.cols; ++x) {
            cv::Vec3f& pixel = processed_img.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                pixel[c] = (pixel[c] - mean[c]) / std_dev[c];
            }
        }
    }    
    std::vector<cv::Mat> channels(3);
    cv::split(processed_img, channels);

    std::vector<float> output;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < channels[c].rows; ++y) {
            for (int x = 0; x < channels[c].cols; ++x) {
                output.push_back(channels[c].at<float>(y, x));
            }
        }
    }

    return output;
}

std::string  Inceptionv1Inference(const std::string& model_path, cv::Mat& image) {
    // Load labels
    std::string label_file_path = "../../data/label.txt";  
    std::vector<std::string> labels = read_imagenet_labels(label_file_path);

    // Initialize ONNX runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inceptionv1");
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
    
    // Run preprocess
    std::vector<float> inputTensor = preprocess(image, inputWidth, inputHeight);
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, inputHeight, inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, inputTensor.data(), inputTensor.size(), input_shape.data(), input_shape.size());

    // Run inference
    std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, num_inputs_, output_node_names_.data(), output_node_names_.size());
    
    // Get output tensor data
    float* output_data = outputs[0].GetTensorMutableData<float>();
    

    // Get predicted class
    int num_classes = output_dims[1];
    int predicted_class = 0;
    float max_score = output_data[0];
    for (int i = 1; i < num_classes; ++i) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            predicted_class = i;
        }
    }

    return labels[predicted_class];
}