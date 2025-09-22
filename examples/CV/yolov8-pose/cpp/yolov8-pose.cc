#include "yolov8-pose.h"
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"



// Image preprocess
cv::Mat Preprocess(
    const cv::Mat& image, int inputWidth , int inputHeight ) {

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
                


    return cv::dnn::blobFromImage(padded_image, 1.0/255.0, 
                                 cv::Size(inputWidth, inputHeight), 
                                 cv::Scalar(0, 0, 0), false, false, CV_32F);

}

Letterbox_t ComputeLetterbox(const cv::Mat& image, int dst_width, int dst_height) {
    // valid check
    if (image.empty() or dst_width <= 0 or dst_height <= 0) {
        std::cerr << "Error: Input image is empty or size is invalid!" << std::endl;
        return {};
    }

    // ROI
    int orig_width = image.cols;
    int orig_height = image.rows;
    float scale_ratio = fmin(static_cast<float>(dst_width) / static_cast<float>(orig_width), static_cast<float>(dst_height) / static_cast<float>(orig_height));
    int scaled_width = static_cast<int>(orig_width * scale_ratio);
    int scaled_height = static_cast<int>(orig_height * scale_ratio);

    int offset_width = (dst_width - scaled_width) / 2;
    int offset_height = (dst_height - scaled_height) / 2;

    Letterbox_t letterbox;
    letterbox.scaled_width = scaled_width;
    letterbox.scaled_height = scaled_height;
    letterbox.offset_width = offset_width;
    letterbox.offset_height = offset_height;
    letterbox.scale_ratio = scale_ratio;

    return letterbox;
}

void GetMapXY(const cv::Mat& src, cv::Mat& map_x, cv::Mat& map_y, Letterbox_t letterbox) {
    if (!map_x.empty() or !map_y.empty()) {
        std::cerr << "map_x and map_y should both be empty" << std::endl;
    }
    int src_width = src.cols;
    int src_height = src.rows;
    cv::Mat map_x_copy(letterbox.scaled_height, letterbox.scaled_width, CV_32FC1, cv::Scalar(-1));
    cv::Mat map_y_copy(letterbox.scaled_height, letterbox.scaled_width, CV_32FC1, cv::Scalar(-1));

    for (int h = 0; h < letterbox.scaled_height; h++) {
        for (int w = 0; w < letterbox.scaled_width; w++) {
            map_x_copy.at<float>(h, w) = w / letterbox.scale_ratio;
            map_y_copy.at<float>(h, w) = h / letterbox.scale_ratio;
        }
    }

    map_x_copy.copyTo(map_x);
    map_y_copy.copyTo(map_y);
}

// Draw results
void DrawResults(cv::Mat& image, const std::vector<Object>& dets) {        
    // Keypoint connections for pose estimation
    std::vector<std::pair<int, int>> kp_connections = {
        {16, 14}, {14, 12}, {15, 13}, {13, 11}, {12, 11},  
        {5, 7}, {7, 9}, {6, 8}, {8, 10},  
        {5, 6}, {5, 11}, {6, 12},  
        {11, 13}, {12, 14},  
        {0, 1}, {0, 2}, {1, 3}, {2, 4},  
        {0, 5}, {0, 6},  
        {3, 5}, {4, 6}  
    };
    
    // Drawing parameters
    cv::Scalar box_color(0, 255, 0);      // Green for bounding box
    cv::Scalar kp_color(255, 0, 0);       // Blue for keypoints
    int line_thickness = 2;
    int kp_radius = 5;
    
    for (const auto& det : dets) {        
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);

        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, line_thickness);        

        // Draw keypoints
        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            const auto& kp = det.keypoints[i];
            if (kp.visibility < point_confidence_threshold) {
                continue;  // Skip invisible keypoints
            }
            // Draw keypoint circle
            cv::circle(image, cv::Point(kp.x, kp.y), kp_radius, kp_color, -1);
        }

        // Draw keypoint connections    
        for (const auto& connection : kp_connections) {
            int start_idx = connection.first;
            int end_idx = connection.second;
            
            // Check bounds
            if (start_idx >= static_cast<int>(det.keypoints.size()) || 
                end_idx >= static_cast<int>(det.keypoints.size())) {
                continue;
            }
            
            const auto& start_kp = det.keypoints[start_idx];
            const auto& end_kp = det.keypoints[end_idx];
            
            // Only draw line if both keypoints are visible
            if (start_kp.visibility < point_confidence_threshold || 
                end_kp.visibility < point_confidence_threshold) {
                continue;
            }
            
            cv::line(image, 
                    cv::Point(start_kp.x, start_kp.y), 
                    cv::Point(end_kp.x, end_kp.y), 
                    kp_color, line_thickness);
        }

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
    std::vector<Object> sorted_dets = dets;

    // Sort detections by score in descending order
    std::sort(sorted_dets.begin(), sorted_dets.end(), [](const Object& a, const Object& b) {
        return a.score > b.score;
    });

    while (!sorted_dets.empty()) {
        // Keep the detection with the highest score
        final_dets.push_back(sorted_dets[0]);

        // If only one detection remains, we're done
        if (sorted_dets.size() == 1) {
            break;
        }

        std::vector<Object> new_sorted_dets;
        
        // Iterate through the rest of the detections
        for (size_t i = 1; i < sorted_dets.size(); ++i) {
            float iou = Calculate_Iou(final_dets.back(), sorted_dets[i]);
            // Only keep detections that don't have high overlap with the current best detection
            if (iou < iou_threshold) {
                new_sorted_dets.push_back(sorted_dets[i]);
            }
        }
        sorted_dets = new_sorted_dets;
    }
    
    return final_dets;
}


// Postprocess output tensor to get detection results
std::vector<Object> Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, int des_width, int des_height) {
    std::vector<Object> objects;            
    float ratio = std::min(static_cast<float>(des_width) / static_cast<float>(input_size.width), static_cast<float>(des_height) / static_cast<float>(input_size.height));
    int unpad_w = std::round(input_size.width * ratio);
    int unpad_h = std::round(input_size.height * ratio);

    float dw = (des_width - unpad_w) / 2.0;
    float dh = (des_height - unpad_h) / 2.0;
    

        
    for (int j = 0; j < anchors; ++j) {        
        if (output[4*anchors + j] > conf_threshold) {
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
            obj.score = output[4*anchors + j] ;
            for(int k = 5; k < 56; ++k) {
                obj.source_keypoints.push_back(output[k*anchors + j]);
            }
            objects.push_back(obj);

        }
    }        
    
    std::vector<Object> objects_nms = Nms(objects, iou_threshold);
    for (auto& obj : objects_nms) {
        for(int k = 0; k < 17; ++k) {
            KeyPoint key_point;
            key_point.x = int((obj.source_keypoints[k*3]-dw)/ratio);
            key_point.y = int((obj.source_keypoints[k*3+1]-dh)/ratio);
            key_point.visibility = obj.source_keypoints[k*3+2];
            obj.keypoints.push_back(key_point);
        }
    }
    return objects_nms;

}



cv::Mat Yolov8PoseInference(cv::Mat& image, const std::string& modelPath) {

    // Initialize ONNX runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8PoseInference");
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

    #ifdef USE_OPENCL
        Letterbox_t letterbox = ComputeLetterbox(image, inputWidth, inputHeight);
        // mapx & mapy
        cv::Mat map_x, map_y;
        GetMapXY(image, map_x, map_y, letterbox);

        // Remap setting
        cv::Mat kernel_in;
        cv::Mat kernel_out;
        std::string kernel_file_path = "../../../../../third-party/opencl/yolo-process/remap.cl";
        std::string kernel_name = "remap_split";
        Remap remapper(kernel_file_path, kernel_name, image.cols, image.rows, map_x, map_y, inputWidth, inputHeight, kernel_in, kernel_out);
        image.copyTo(kernel_in);
        remapper.remap();
    #else
        cv::Mat inputTensor = Preprocess(image, inputWidth, inputHeight);    
    #endif    
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, inputHeight, inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    #ifdef USE_OPENCL
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, kernel_out.ptr<float>(), kernel_out.total(), input_shape.data(), input_shape.size());
    #else
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(inputTensor.data), 3 * inputHeight * inputWidth, input_shape.data(), input_shape.size());
    #endif

    // Run inference
    std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), output_node_names_.size());
    
    
    // Get output data
    float* dets_data = outputs[0].GetTensorMutableData<float>();


    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();    
    int offset = dets_dims[1];
    int anchors = dets_dims[2];
    //Run postprocess
    std::vector<Object> detected_objects = Postprocess(image.size(),dets_data, anchors, offset, inputWidth, inputHeight);

    // Draw results
    DrawResults(image, detected_objects);

    return image;
}