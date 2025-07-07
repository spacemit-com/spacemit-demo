#include "yolov11.h"
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"



// Image preprocess
cv::Mat Preprocess(
    const cv::Mat& image, int inputWidth, int inputHeight) {

    if (image.empty() || inputWidth <= 0 || inputHeight <= 0) {
        std::cerr << "Error: Input image is empty or destination size is invalid." << std::endl;
        return cv::Mat();
    }
    
    // Get the original shape of the image
    int height = image.rows;
    int width = image.cols;
    cv::Vec3b pad_color(0, 0, 0);

    // Calculate the scaling ratio
    double r = std::min(static_cast<double>(inputHeight) / height, static_cast<double>(inputWidth) / width);
    // Compute padding
    double new_unpad_h = std::round(height * r);
    double new_unpad_w = std::round(width * r);
    int dw = inputWidth - static_cast<int>(new_unpad_w);
    int dh = inputHeight - static_cast<int>(new_unpad_h);
    dw /= 2;
    dh /= 2;

    // Resize the image if necessary
    cv::Mat resized_image;
    if (height != new_unpad_h || width != new_unpad_w) {
        cv::resize(image, resized_image, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_image = image.clone();
    }

    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // Add border
    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));
    cv::Mat padded_image;

    cv::copyMakeBorder(rgb_image, padded_image, top, bottom, left, right, cv::BORDER_CONSTANT, pad_color);
                

    cv::Mat blob;
    blob = cv::dnn::blobFromImage(padded_image,1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(0, 0, 0), false, false,CV_32F);

    return blob;

}

// Draw results
void DrawResults(cv::Mat& image, const std::vector<Object>& dets, const std::vector<std::string>& labels) {        
    int image_h = image.rows;
    int image_w = image.cols;

    for (const auto& det : dets) {        
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);

        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);        

        // Draw label and score
        std::string labelText = labels[det.class_id] + ": " + std::to_string(det.score).substr(0, 4);
        cv::putText(image, labelText, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);        
    }
}

// Compute IOU between two detections
float Calculate_Iou(const Object& det1, const Object& det2) {
    float x1_inter = std::max(det1.x1, det2.x1);
    float y1_inter = std::max(det1.y1, det2.y1);
    float x2_inter = std::min(det1.x2, det2.x2);
    float y2_inter = std::min(det1.y2, det2.y2);

    float width_inter = std::max(0.0f, x2_inter - x1_inter);
    float height_inter = std::max(0.0f, y2_inter - y1_inter);
    float area_inter = width_inter * height_inter;

    float area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1);
    float area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1);
    float area_union = area1 + area2 - area_inter;

    if (area_union == 0) {
        return 0;
    }
    return area_inter / area_union;
}

// Non-maximum suppression
std::vector<Object> Nms(const std::vector<Object>& dets, float iou_threshold = 0.45) {
    if (dets.empty()) {
        return std::vector<Object>();
    }

    std::vector<Object> final_dets;

    // Divide detections into different classes
    std::vector<int> unique_labels;
    for (const auto& det : dets) {
        if (std::find(unique_labels.begin(), unique_labels.end(), det.class_id) == unique_labels.end()) {
            unique_labels.push_back(det.class_id);
        }
    }

    for (int label : unique_labels) {
        std::vector<Object> dets_class;
        // Get detections for the current class
        for (const auto& det : dets) {
            if (det.class_id == label) {
                dets_class.push_back(det);
            }
        }

        // Sort detections by score
        std::sort(dets_class.begin(), dets_class.end(), [](const Object& a, const Object& b) {
            return a.score > b.score;
        });

        std::vector<Object> keep;
        while (!dets_class.empty()) {
        
            keep.push_back(dets_class[0]);
            if (dets_class.size() == 1) {
                break;
            }

            std::vector<Object> new_dets_class;
        
            for (size_t i = 1; i < dets_class.size(); ++i) {
                float iou = Calculate_Iou(keep.back(), dets_class[i]);
                if (iou < iou_threshold) {
                    new_dets_class.push_back(dets_class[i]);
                }
            }
            dets_class = new_dets_class;
        }
        
        final_dets.insert(final_dets.end(), keep.begin(), keep.end());
    }

    return final_dets;
}

// Read labels from file
std::vector<std::string> readLabels(const std::string& labelFilePath) {
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

// Postprocess output tensor to get detection results
std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, float conf_threshold, float iou_threshold, int des_width,int des_height) {
    std::vector<Object> objects;            
    float ratio = std::min(static_cast<float>(des_width) / static_cast<float>(input_size.width), static_cast<float>(des_height) / static_cast<float>(input_size.height));
    int unpad_w = std::round(input_size.width * ratio);
    int unpad_h = std::round(input_size.height * ratio);

    float dw = (des_width - unpad_w) / 2.0;
    float dh = (des_height - unpad_h) / 2.0;
    
    for (int j = 0; j < anchors; ++j) {
        float max_score = -1.0f;
        int max_index = -1;
        for (int prob = 4; prob < offset; ++prob) {
            // Get the score for the current anchor and class
            int index = prob * anchors + j;
            float score = output[index];
            if (score > max_score) {                
                max_score = score;
                max_index = prob;
            }

        }        
        
        if (max_score > conf_threshold) {
            // Decoder
            float half_width = output[2 * anchors + j] / 2;
            float half_height = output[3 * anchors + j] / 2;            
            int x1 = (output[j] - half_width - dw) / ratio;        
            x1 = std::max(0,x1);
            int y1 = (output[anchors + j] - half_height - dh) / ratio;
            y1 = std::max(0,y1);
            int x2 = (output[j] + half_width - dw) / ratio;
            x2 = std::max(0,x2);
            int y2 = (output[anchors + j] + half_height - dh) / ratio;
            y2 = std::max(0,y2);
           
            Object obj;
            obj.x1 = x1;
            obj.y1 = y1;
            obj.x2 = x2;
            obj.y2 = y2;
            obj.class_id = max_index - 4;
            obj.score = max_score;
            objects.push_back(obj);

        }
    }        
    
    return Nms(objects, iou_threshold);

}



cv::Mat Yolov11Inference(cv::Mat& image, const std::string& modelPath, const std::string& labelFilePath, float conf_threshold, float iou_threshold) {
    // Load labels
    std::vector<std::string> labels = readLabels(labelFilePath);

    // Initialize ONNX runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11Inference");
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
    
    
    // Get output data
    float* dets_data = outputs[0].GetTensorMutableData<float>();


    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();    
    int offset = dets_dims[1];
    int anchors = dets_dims[2];
    
    //Run postprocess
    std::vector<Object> detected_objects = Postprocess(image.size(),dets_data, anchors, offset, conf_threshold, iou_threshold, inputWidth, inputHeight);

    // Draw results
    DrawResults(image, detected_objects, labels);

    return image;
}