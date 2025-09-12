#include "yolov8-seg.h"

cv::Mat preprocess(
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
                

    return cv::dnn::blobFromImage(padded_image,1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(0, 0, 0), false, false,CV_32F);
}



// 计算两个检测框的 IoU
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

    return (area_union > 0) ? area_inter / area_union : 0.0f;
}


std::vector<Object> Nms(const std::vector<Object>& dets) {
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
            dets_class = std::move(new_dets_class);
        }
        
        final_dets.insert(final_dets.end(), keep.begin(), keep.end());
    }

    return final_dets;
}


// 读取标签文件
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




// Sigmoid 函数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}



cv::Mat visualize_results(cv::Mat &scaled_image, const std::vector<Object>& objects, float *output_proto, int dw,int dh, std::vector<int64_t> proto_dims, int des_width, int des_height){
    cv::Mat image = scaled_image.clone();
    float seg_threshold = 0.5;
    float alpha = 0.5;

    
    Eigen::Map<Eigen::Matrix<float, 32, Eigen::Dynamic, Eigen::RowMajor>> 
    proto_matrix(output_proto, 32, proto_dims[2] * proto_dims[3]);
    
    
    // Create mask matrix manually since detect_masks is a vector of vectors
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mask_matrix(objects.size(), proto_dims[1]);
    for(int i = 0; i < objects.size(); i++)
    {
        for(int j = 0; j < proto_dims[1]; j++)
        {
            mask_matrix(i,j) = objects[i].detect_masks[0][j];
        }
    }

    Eigen::MatrixXf seg_matrix = mask_matrix * proto_matrix;        
    seg_matrix = seg_matrix.unaryExpr([](float x) { return sigmoid(x); });

      
    Eigen::MatrixXf transposed_eigen_matrix = seg_matrix.transpose();
    std::vector<float> flat_data(transposed_eigen_matrix.data(), transposed_eigen_matrix.data() + transposed_eigen_matrix.rows() * transposed_eigen_matrix.cols());
    std::vector<cv::Mat> mask_mats;
    for (int i = 0; i < seg_matrix.rows(); ++i) {        
        int start_index = i * (proto_dims[2] * proto_dims[3]);

        // Create a pointer to the beginning of the current sub-matrix data
        float* data_ptr = &flat_data[start_index];

        // Create a cv::Mat with row-major order using the data pointer
        cv::Mat mask_mat(proto_dims[2], proto_dims[3], CV_32FC1, data_ptr);        
        mask_mats.push_back(mask_mat.clone());
    }        
    
    for (int i = 0; i < mask_mats.size(); i++){
        // Resize the segmentation image
        cv::Mat resize_mask;        
        cv::resize(mask_mats[i], resize_mask, cv::Size(des_width, des_height), 0, 0, cv::INTER_LINEAR);
        // Get color index and color
        int color_index = objects[i].class_id % src_colors.size();
        cv::Scalar color = src_colors[color_index];
        // Extract bounding box coordinates
        int x1 = objects[i].x1;
        int y1 = objects[i].y1;
        int x2 = objects[i].x2;
        int y2 = objects[i].y2;
        // Crop the resized image to the original size
        // resize_mask = resize_mask(cv::Rect(dw,dh,des_width,des_height));
        cv::Mat crop_resize_mask = resize_mask(cv::Rect(dw,dh,des_width-dw,des_height-dh));
        
        // Extract the region of interest (ROI)
        cv::Mat roi = crop_resize_mask(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        
        // Create a mask based on the threshold
        cv::Mat mask;
        cv::compare(roi, seg_threshold, mask, cv::CMP_GT);
        
        // Create a color mask
        cv::Mat color_mask(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        color_mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo( color, mask);
        
        // Overlay the color mask onto the original image
        cv::addWeighted(image, 1.0, color_mask, alpha, 0.0, image);

    }

    std::vector<std::string> labels = readLabels(labelFilePath);
    for(int i = 0; i < objects.size(); i++)
    {
        cv::rectangle(image, cv::Point(objects[i].x1, objects[i].y1), cv::Point(objects[i].x2, objects[i].y2), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, labels[objects[i].class_id], cv::Point(objects[i].x1, objects[i].y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }
    
    return image;

}

std::array<float, 4> Dfl(const float* boxes, int anchor_idx, int anchors ) {
    std::array<float, 4> xywh = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        float exp_sum = 0.0f;

        size_t offset = i * dfl_len * anchors + anchor_idx; // 起始位置
        float exp_dfl[dfl_len];
        for (int dfl_idx = 0; dfl_idx < dfl_len; dfl_idx++) {
            exp_dfl[dfl_idx] = exp(boxes[offset]);
            exp_sum += exp_dfl[dfl_idx];
            offset += anchors;
        }

        offset = i * dfl_len * anchors + anchors; // reset
        for (int dfl_idx = 0; dfl_idx < dfl_len; dfl_idx++) {
            xywh[i] += (exp_dfl[dfl_idx] / exp_sum) * dfl_idx;
            offset += anchors;
        }
    }

    return xywh;
}   

void Get_Dets(const cv::Mat& image,
              const float* boxes, 
              const float* scores, 
              const float* score_sum, 
              std::vector<int64_t> dims, 
              int tensor_width, 
              int tensor_height, 
              std::vector<Object>& objects,
              const float* seg_part,
              int pad_w,
              int pad_h
) {
    int grid_w = static_cast<int>(dims[2]);
    int grid_h = static_cast<int>(dims[3]);
    int anchors_per_branch = grid_w * grid_h;
    float scale_w = static_cast<float>(tensor_width) / static_cast<float>(grid_w);
    float scale_h = static_cast<float>(tensor_height) / static_cast<float>(grid_h);

    for (int anchor_idx = 0; anchor_idx < anchors_per_branch; anchor_idx++) {
        if (score_sum[anchor_idx] < conf_threshold) {
            continue;
        }

        // get max score and class
        float max_score = -1.0f;
        int classId = -1;
        for (int class_idx = 0; class_idx < classNum; class_idx++) {
            size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
            if ((scores[score_offset] > conf_threshold) & (scores[score_offset] > max_score)) { 
                max_score = *(scores + score_offset);
                classId = class_idx;                
            } else {
                continue;
            }
        }
        if (classId >= 0) { // detect object
            std::array<float, 4> xywh = Dfl(boxes, anchor_idx, anchors_per_branch);
            int h_idx = anchor_idx / grid_w;
            int w_idx = anchor_idx % grid_w;
            std::vector<float> seg_part_vector;
            for (int i = 0; i < 32 ; i++) {
                seg_part_vector.push_back(seg_part[i * anchors_per_branch + anchor_idx]);
            }

            Object object;
            object.detect_masks.push_back(seg_part_vector);
            object.x1 = ((w_idx - xywh[0] + 0.5f) * scale_w - pad_w); //  no scale2orign,keep the sacle size;
            object.y1 = ((h_idx - xywh[1] + 0.5f) * scale_h - pad_h);
            object.x2 = ((w_idx + xywh[2] + 0.5f) * scale_w - pad_w);
            object.y2 = ((h_idx + xywh[3] + 0.5f) * scale_h - pad_h);
        
            object.class_id = classId;
            object.score = max_score;
            objects.push_back(object);
        } else { // no object
            continue;
        }
    }
  
};


cv::Mat Postprocess(cv::Mat &image, std::vector<Ort::Value>& outputs, size_t output_num, const int inputWidth, const int inputHeight, std::vector<Object> &objects) {
    float* output_proto = outputs[12].GetTensorMutableData<float>();    
    std::vector<int64_t> proto_dims = outputs[12].GetTensorTypeAndShapeInfo().GetShape();
    
    int orig_height = image.rows;
    int orig_width = image.cols;
    float scale2orign = fmin(static_cast<float>(inputHeight) / static_cast<float>(orig_width), static_cast<float>(inputWidth) / static_cast<float>(orig_height));
    int pad_h = static_cast<int>((inputWidth - orig_height * scale2orign) / 2);
    int pad_w = static_cast<int>((inputHeight - orig_width * scale2orign) / 2);
    float new_w = orig_width * scale2orign;
    float new_h = orig_height * scale2orign;

    // Calculate dw and dh for visualization
    int dw = static_cast<int>((inputWidth - new_w) / 2);
    int dh = static_cast<int>((inputHeight - new_h) / 2);

    for (int i = 0; i < int(output_num / branch_element ); i++) {        
        const float* boxes = outputs[i * 3].GetTensorMutableData<float>(); // get data
        const float* scores = outputs[i * 3 + 1].GetTensorMutableData<float>();
        const float* score_sum = outputs[i * 3 + 2].GetTensorMutableData<float>();
        const float* seg_part = outputs[9 + i].GetTensorMutableData<float>();
        std::vector<int64_t> dims = outputs[i * 3].GetTensorTypeAndShapeInfo().GetShape(); // 输出的tensor类型和维度                    
        Get_Dets(image, boxes, scores, score_sum, dims, inputHeight, inputWidth, objects, seg_part, pad_w, pad_h); // swap width and height        
    }
    std::vector<Object> Nms_objects = Nms(objects);
    cv::Mat scaled_image;
    cv::resize(image, scaled_image, cv::Size(new_w, new_h));

    cv::Mat result = visualize_results(scaled_image, Nms_objects, output_proto, dw, dh, proto_dims, inputWidth, inputHeight);

    
    return result;

}







cv::Mat Yolov8SegInference(const cv::Mat& src_image, const std::string& modelPath) {
        
    // 初始化 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-Seg Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    SessionOptionsSpaceMITEnvInit(session_options);
    // 加载 ONNX 模型
    Ort::Session session_(env, modelPath.c_str(), session_options);

    // 获取输入和输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*> input_node_names_;
    std::vector<std::string> input_names_;
    // input names initial and build
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

    
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_names_;

    // output names initial and build
    size_t num_outputs_;
    num_outputs_ = session_.GetOutputCount();
    output_node_names_.resize(num_outputs_);
    output_names_.resize(num_outputs_, "");    
    for (size_t i = 0; i < num_outputs_; ++i) {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names_[i].append(output_name.get());        
        output_node_names_[i] = output_names_[i].c_str();
    }

    
    cv::Mat image = src_image.clone();        
    cv::Mat inputTensor = preprocess(image, inputWidth, inputHeight);

    // 创建输入张量
    std::vector<int64_t> input_shape = {1, 3, inputHeight, inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(inputTensor.data), 3 * inputHeight * inputWidth, input_shape.data(), input_shape.size());

    // 进行推理
    std::vector<Ort::Value> outputs;    
    outputs = session_.Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), output_node_names_.size());


    // 获取输出数据
    float* dets_data = outputs[0].GetTensorMutableData<float>();
    float* proto_data = outputs[1].GetTensorMutableData<float>();

    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();    

    auto dets_tensor_info1 = outputs[1].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> proto_dims = dets_tensor_info1.GetShape();    
    
    
    // Declare objects vector
    std::vector<Object> objects;
    
    cv::Mat result_image = Postprocess(image, outputs, num_outputs_, inputWidth, inputHeight, objects);
    
        
    return result_image;
}