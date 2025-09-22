#include"yoloe.h"
#include <cmath>
#include <chrono>
#include <iostream>
YOLOE::YOLOE(const std::string& yolo_world_model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOE"),
    session1_(nullptr)
{
    Ort::SessionOptions  session_options_1;
    session_options_1.SetIntraOpNumThreads(4);
    //SessionOptionsSpaceMITEnvInit(session_options_1);
    session1_ = std::make_unique<Ort::Session>(env_, yolo_world_model_path.c_str(),session_options_1);

    num_inputs = session1_->GetInputCount();
    input1_names_.resize(num_inputs);
    input1_names.resize(num_inputs, "");

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session1_->GetInputNameAllocated(i, allocator_);
        input1_names[i].append(input_name.get());
        input1_names_[i] = input1_names[i].c_str();

        Ort::TypeInfo input_type_info = session1_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape(); // 获得输入维度
        if(input_dims.size()==4)
        {
            input_image_dims=input_dims;
        }
        else
        {
            input_text_dims=input_dims;
        }

    }
    num_outputs = session1_->GetOutputCount();
    output1_names_.resize(num_outputs);
    output1_names.resize(num_outputs, "");    
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session1_->GetOutputNameAllocated(i, allocator_);
        output1_names[i].append(output_name.get());
        output1_names_[i] = output1_names[i].c_str();
    }

    text_feature_data_.reserve(input_text_dims[1]*input_text_dims[2]); 
    text_feature_data_.assign(input_text_dims[1]*input_text_dims[2], 0.0f);


}
YOLOE::~YOLOE()
{

}
void YOLOE::Preprocess(const cv::Mat& image, cv::Mat& blob_image)
{
    cv::Mat canvas=pad_image(cv::Rect(letterbox.offset_width, letterbox.offset_height, letterbox.scaled_width, letterbox.scaled_height));
    cv::resize(image, canvas, cv::Size(letterbox.scaled_width, letterbox.scaled_height), 0, 0, cv::INTER_LINEAR);
    // normalize
    blob_image = cv::dnn::blobFromImage(pad_image, 1.0 / 255.0, cv::Size(pad_image.cols, pad_image.rows), cv::Scalar(0, 0, 0), true, false, CV_32F);

}

std::vector<Object> YOLOE::Postprocess(const cv::Size& input_size, const float* output, int anchors, int offset, float conf_threshold, float iou_threshold, int des_width,int des_height)
{
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
std::vector<Object> YOLOE::Nms(const std::vector<Object>& dets, float iou_threshold)
{
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

float YOLOE::Calculate_Iou(const Object& det1, const Object& det2)
{
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
std::vector<std::string> YOLOE::ReadLabels(const std::string& labelFilePath)
{
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
void YOLOE::DrawResults(cv::Mat& image, const std::vector<Object>& dets, std::vector<std::string>& labels)
{
    int image_h = image.rows;
    int image_w = image.cols;    

    for (const auto& det : dets) 
    {        
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

void YOLOE::init_yoloe_model(std::vector<std::vector<float>>clipdata,const cv::Mat& image,float conf_threshold_, float iou_threshold_)
{
    int orig_width = image.cols;
    int orig_height = image.rows;
    int dst_height=input_image_dims[2];
    int dst_width=input_image_dims[3];
    pad_image=cv::Mat(dst_height, dst_width, image.type(), cv::Scalar(0, 0, 0));
     
    float scale = fmin(static_cast<float>(dst_width) / static_cast<float>(orig_width), static_cast<float>(dst_height) / static_cast<float>(orig_height));
    int scaled_width = static_cast<int>(orig_width * scale);
    int scaled_height = static_cast<int>(orig_height * scale);
    int offset_width = (dst_width - scaled_width) / 2;
    int offset_height = (dst_height - scaled_height) / 2;

    letterbox.scaled_width = scaled_width;
    letterbox.scaled_height = scaled_height;
    letterbox.offset_width = offset_width;
    letterbox.offset_height = offset_height; 

    for(int i=0;i<clipdata.size();i++)
    {
        std::vector<float>data=clipdata[i];
        size_t copy_size=data.size();
        std::copy(data.begin(), data.begin() + copy_size, text_feature_data_.begin()+i*copy_size); 

    }

    conf_threshold=conf_threshold_;
    iou_threshold=iou_threshold_;
}
cv::Mat YOLOE::inference_yoloe_model(cv::Mat& image)
{
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(2);
    cv::Mat inputTensor;
    Preprocess(image, inputTensor);

    std::vector<int64_t> input_image_shape = {1, 3, input_image_dims[2], input_image_dims[3]};
    auto memory_image_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // 开辟内存
    Ort::Value input_image_tensor = Ort::Value::CreateTensor<float>(memory_image_info, inputTensor.ptr<float>(), inputTensor.total(), input_image_shape.data(), input_image_shape.size());

    std::vector<int64_t> input_text_shape = {1, input_text_dims[1], input_text_dims[2]};
    auto memory_text_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // 开辟内存
    Ort::Value input_text_tensor = Ort::Value::CreateTensor<float>(memory_text_info, text_feature_data_.data(), text_feature_data_.size(), input_text_shape.data(), input_text_shape.size());

    input_tensors.push_back(std::move(input_image_tensor));
    input_tensors.push_back(std::move(input_text_tensor));

    std::vector<Ort::Value> outputs= session1_->Run(Ort::RunOptions{nullptr}, input1_names_.data(), input_tensors.data(), num_inputs, output1_names_.data(), output1_names_.size());

    // Get output data
    float* dets_data = outputs[0].GetTensorMutableData<float>();

    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();    
    int offset = dets_dims[1];
    int anchors = dets_dims[2];

    //Run postprocess
    std::vector<Object> detected_objects = Postprocess(image.size(),dets_data, anchors, offset, conf_threshold, iou_threshold, input_image_dims[3], input_image_dims[2]);

    // Draw results
    DrawResults(image, detected_objects, labels);

    return image;

}
