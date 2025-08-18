#include "yolov5.h"
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"



// Image preprocess
cv::Mat Preprocess(const cv::Mat& image, int inputWidth, int inputHeight) {
    cv::Mat blob;
    blob = cv::dnn::blobFromImage(image,1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(0, 0, 0), true, false,CV_32F);

    return blob;
}

// Draw results
void DrawResults(cv::Mat& image, const std::vector<std::vector<float>>& dets, const std::vector<float>& scores, int* labels_pred, const std::vector<std::string>& labels) {    
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& det = dets[i];
        float score = scores[i];        
        if (score > conf_threshold) {
            int class_id = labels_pred[i];
            int x1 = static_cast<int>(det[0]);
            int y1 = static_cast<int>(det[1]);
            int x2 = static_cast<int>(det[2]);
            int y2 = static_cast<int>(det[3]);                        
            std::string label = labels[class_id];
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(image, label + ": " + std::to_string(score), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }
    }    
}


// Read labels from file
std::vector<std::string> ReadLabels() {
    std::vector<std::string> labels;
    std::ifstream labelFile(labelFilePath);
    if (labelFile.is_open()) {
        std::string line;
        while (std::getline(labelFile, line)) {
            labels.push_back(line);
        }
        labelFile.close();
    }
    return labels;
}



cv::Mat Yolov5Inference(cv::Mat& image, const std::string& modelPath) {
    // Load labels
    std::vector<std::string> labels = ReadLabels();

    // Initialize ONNX runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    SessionOptionsSpaceMITEnvInit(session_options);
    // Load ONNX model
    Ort::Session session_(env, modelPath.c_str(), session_options);

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

    // Run preprocess
    cv::Mat inputTensor = Preprocess(image, inputWidth, inputHeight);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, inputHeight, inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(inputTensor.data), 3 * inputHeight * inputWidth, input_shape.data(), input_shape.size());

    // Run inference
    std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), output_node_names_.size());

    float* dets_data = outputs[0].GetTensorMutableData<float>();
    int* labels_pred_data = outputs[1].GetTensorMutableData<int>();    

    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();    
    size_t num_detections = dets_dims[1];
    
    // Get detections    
    std::vector<std::vector<float>> dets(num_detections, std::vector<float>(4));
    std::vector<float> scores(num_detections);
    
    for (size_t i = 0; i < num_detections; ++i) {
        for (int j = 0; j < 4; ++j) {
            dets[i][j] = dets_data[i * 5 + j];                    
        }    
        
        scores[i] = dets_data[i * 5 + 4];                
    }
    
    cv::Size image_shape = image.size();
    // Rescale detections to original image size
    float scale_x = static_cast<float>(image_shape.width) / inputWidth;
    float scale_y = static_cast<float>(image_shape.height) / inputHeight;
    for (auto& det : dets) {
        det[0] *= scale_x;
        det[1] *= scale_y;
        det[2] *= scale_x;
        det[3] *= scale_y;
    }

    
    
    // Draw results
    DrawResults(image, dets, scores, labels_pred_data, labels);

    return image;
}