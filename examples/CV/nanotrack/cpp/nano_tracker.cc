#include "NanoTracker.h"
#include <cmath>
#include <chrono>
#include <iostream>

NanoTracker::NanoTracker(const std::string& onnx_backbone1_path,
                         const std::string& onnx_backbone2_path,
                         const std::string& onnx_ban_head_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "NanoTracker"),
      session1_(nullptr),
      session2_(nullptr),
      session3_(nullptr) {
    // 初始化参数
    score_size_ = 16;
    cls_out_channels_ = 2;
    point_stride_ = 16;
    track_context_amount_ = 0.5;
    track_exemplar_size_ = 127;
    track_instance_size_ = 255;
    track_penalty_k_ = 0.148;
    track_window_influence_ = 0.462;
    track_lr_ = 0.390;

    Ort::SessionOptions  session_options_1;

    Ort::SessionOptions session_options_2;
    session_options_2.SetIntraOpNumThreads(4);
    SessionOptionsSpaceMITEnvInit(session_options_2);

    Ort::SessionOptions session_options_3;       
    session_options_3.SetIntraOpNumThreads(1);
    SessionOptionsSpaceMITEnvInit(session_options_3);
    
    

    
    session1_ = std::make_unique<Ort::Session>(env_, onnx_backbone1_path.c_str(),session_options_1);
    session2_ = std::make_unique<Ort::Session>(env_, onnx_backbone2_path.c_str(), session_options_2);
    session3_ = std::make_unique<Ort::Session>(env_, onnx_ban_head_path.c_str(),session_options_3);
 


    // 获取输入和输出信息    
    // input names initial and build
    num_inputs = session1_->GetInputCount();
    input1_names_.resize(num_inputs);
    input1_names.resize(num_inputs, "");
    
    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session1_->GetInputNameAllocated(i, allocator_);
        input1_names[i].append(input_name.get());
        input1_names_[i] = input1_names[i].c_str();
    } 
    num_outputs = session1_->GetOutputCount();
    output1_names_.resize(num_outputs);
    output1_names.resize(num_outputs, "");
    
    for (size_t i = 0; i < num_outputs; ++i) {        
        auto output_name = session1_->GetOutputNameAllocated(i, allocator_);        
        output1_names[i].append(output_name.get());
        output1_names_[i] = output1_names[i].c_str();
    }
    
    
    num_inputs = session2_->GetInputCount();
    input2_names_.resize(num_inputs);
    input2_names.resize(num_inputs, "");

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session2_->GetInputNameAllocated(i, allocator_);
        input2_names[i].append(input_name.get());
        input2_names_[i] = input2_names[i].c_str();
    }
    num_outputs = session2_->GetOutputCount();    
    output2_names_.resize(num_outputs);
    output2_names.resize(num_outputs, "");
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session2_->GetOutputNameAllocated(i, allocator_);
        output2_names[i].append(output_name.get());
        output2_names_[i] = output2_names[i].c_str();
    }

    num_inputs = session3_->GetInputCount();
    input3_names_.resize(num_inputs);
    input3_names.resize(num_inputs, "");

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session3_->GetInputNameAllocated(i, allocator_);
        input3_names[i].append(input_name.get());
        input3_names_[i] = input3_names[i].c_str();
    }
    num_outputs = session3_->GetOutputCount();
    output3_names_.resize(num_outputs);
    output3_names.resize(num_outputs, "");    
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session3_->GetOutputNameAllocated(i, allocator_);
        output3_names[i].append(output_name.get());

        output3_names_[i] = output3_names[i].c_str();
    }
    // 生成 Hanning 窗
    std::vector<float> hanning_window = hanning(score_size_);
    window_ = outer(hanning_window, hanning_window);

    // 生成点
    points = generate_points(point_stride_, score_size_);
}

NanoTracker::~NanoTracker() {
    // 析构函数，目前无需特殊清理
}

void NanoTracker::init(const cv::Mat& img, const cv::Rect2f& bbox) {
    center_pos_ = cv::Point2f(bbox.x + (bbox.width -1) / 2.0, bbox.y + (bbox.height-1) / 2.0);
    
    size_ = cv::Size2f(bbox.width, bbox.height);    
    mean_val = cv::mean(img);
    int r = img.rows;
    int c = img.cols;
    int r_c = std::max(r, c);
    
    padded_image=cv::Mat::zeros(r+r_c, c+r_c, CV_8UC3);
    


   float w_z = size_.width + track_context_amount_ * (size_.width + size_.height);
   float h_z = size_.height + track_context_amount_ * (size_.width + size_.height);
   float s_z = std::sqrt(w_z * h_z);   

   //std::vector<float>  z_crop;
   cv::Mat z_crop = get_subwindow(img, {center_pos_.x,center_pos_.y}, track_exemplar_size_, static_cast<int>(std::round(s_z)), mean_val);
   

   

   std::vector<int64_t> input_shape = {1, 3, track_exemplar_size_, track_exemplar_size_};
   
   Ort::Value input_tensor = create_input_tensor(z_crop, input_shape);
   std::vector<Ort::Value> outputs = session1_->Run(Ort::RunOptions{nullptr}, input1_names_.data(), &input_tensor, 1, output1_names_.data(), output1_names_.size());
   
    // 提取 z_feature_ 的数据并存储到 z_feature_data_ 中
    auto [data, shape] = get_tensor_data<float>(outputs[0]);
    z_feature_data_.assign(data, data + shape[0] * shape[1] * shape[2] * shape[3]);
    z_feature_shape_ = shape;       
        
}

inline float change(float r) {
    return std::max(r, 1.0f / r);
}


inline float sz(float w, float h) {
    float pad = (w + h) * 0.5f;
    return std::sqrt((w + pad) * (h + pad));
}


TrackResult NanoTracker::track(const cv::Mat& img) {
    
    float w_z = size_.width + track_context_amount_ * (size_.width + size_.height);
    float h_z = size_.height + track_context_amount_ * (size_.width + size_.height);
    float s_z = std::sqrt(w_z * h_z);
    float scale_z = track_exemplar_size_ / s_z;
    float s_x = s_z * (track_instance_size_ / track_exemplar_size_);

    
    cv::Mat x_crop = get_subwindow(img, {center_pos_.x,center_pos_.y}, track_instance_size_, static_cast<int>(std::round(s_x)), mean_val);
    std::vector<int64_t> input_shape = {1, 3, track_instance_size_, track_instance_size_};


    Ort::Value input_tensor = create_input_tensor(x_crop, input_shape);
    auto outputs2 = session2_->Run(Ort::RunOptions{nullptr}, input2_names_.data(), &input_tensor, 1, output2_names_.data(), output2_names_.size());

    
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(2);
    // 从 z_feature_data_ 重新创建 Ort::Value
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value z_feature_ = Ort::Value::CreateTensor<float>(memory_info, z_feature_data_.data(), z_feature_data_.size(), z_feature_shape_.data(), z_feature_shape_.size());
    input_tensors.push_back(std::move(z_feature_));
    input_tensors.push_back(std::move(outputs2[0]));

    std::vector<Ort::Value> outputs_final = session3_->Run(Ort::RunOptions{nullptr}, input3_names_.data(), input_tensors.data(), 2, output3_names_.data(), output3_names_.size());

    
    auto [score_data, score_shape] = get_tensor_data<float>(const_cast<Ort::Value&>(outputs_final[0]));
    auto [delta_data, delta_shape] = get_tensor_data<float>(const_cast<Ort::Value&>(outputs_final[1]));

    std::vector<float> score;
    
        
    for(int i = 0; i < score_shape[2]*score_shape[3]; i++)
    {
        float x1 =  score_data[i];
        float x2 =  score_data[i+16*16];
        float max_x = std::max(x1,x2);
        float e_x1 = std::exp(x1-max_x);
        float e_x2 = std::exp(x2-max_x);        
        score.emplace_back(e_x2/(e_x1+e_x2));
    }
        
    // 处理边界框    
    std::vector<std::vector<float>> delta(delta_shape[1], std::vector<float>(delta_shape[2]*delta_shape[3]));
    for(int i = 0; i < delta_shape[1]; i++)
    {
        for(int j = 0; j < delta_shape[2]*delta_shape[3]; j++)
        {
            delta[i][j] = delta_data[i*delta_shape[2]*delta_shape[3]+j];
        }
    }

    size_t num_points = points.size();
    for (size_t i = 0; i < num_points; ++i) {
        delta[0][i] = points[i][0] - delta[0][i]; // x1
        delta[1][i] = points[i][1] - delta[1][i]; // y1
        delta[2][i] = points[i][0] + delta[2][i]; // x2
        delta[3][i] = points[i][1] + delta[3][i]; // y2
    } 
    std::vector<std::vector<float>> pred_bbox = corner2center(delta);

    std::vector<float> s_c(num_points);
    for (int i = 0; i < num_points; ++i) {
        float bbox_w = pred_bbox[2][i];
        float bbox_h = pred_bbox[3][i];
        s_c[i] = change(sz(bbox_w, bbox_h) / sz(size_.width * scale_z, size_.height * scale_z));
    }

    // 2. Aspect ratio penalty
    std::vector<float> r_c(num_points);
    for (int i = 0; i < num_points; ++i) {
        float bbox_w = pred_bbox[2][i];
        float bbox_h = pred_bbox[3][i];
        r_c[i] = change((size_.width / size_.height) / (bbox_w / bbox_h));
    }

    // 3. Penalty and score
    std::vector<float> penalty(num_points);
    std::vector<float> pscore(num_points);
    for (int i = 0; i < num_points; ++i) {
        penalty[i] = std::exp(-(r_c[i] * s_c[i] - 1.0f) * track_penalty_k_);
        pscore[i] = penalty[i] * score[i];
    }

    // 4. Window penalty
    for (int i = 0; i < num_points; ++i) {
        pscore[i] = pscore[i] * (1.0f - track_window_influence_) +
                    window_[i] * track_window_influence_;
    }

    // 5. Find the best index
    auto max_iter = std::max_element(pscore.begin(), pscore.end());
    int best_idx = std::distance(pscore.begin(), max_iter);

    // 6. Calculate bbox
    std::vector<float> bbox(4);
    for (int i = 0; i < 4; ++i) {
        bbox[i] = pred_bbox[i][best_idx] / scale_z;
    }

    float lr = penalty[best_idx] * score[best_idx] * track_lr_;

    float cx = bbox[0] + center_pos_.x;
    float cy = bbox[1] + center_pos_.y;

    // Smooth bbox
    float width = size_.width * (1.0f - lr) + bbox[2] * lr;
    float height = size_.height * (1.0f - lr) + bbox[3] * lr;

    // box_clp
    cx = std::max(0.0f, std::min(cx, static_cast<float>(img.cols)));
    cy = std::max(0.0f, std::min(cy, static_cast<float>(img.rows)));
    width = std::max(10.0f, std::min(width, static_cast<float>(img.cols)));
    height = std::max(10.0f, std::min(height, static_cast<float>(img.rows)));

    center_pos_ = cv::Point2f(cx, cy);
    size_ = cv::Size2f(width, height);

    TrackResult result;
    result.bbox = cv::Rect2f(center_pos_.x - size_.width / 2.0, center_pos_.y - size_.height / 2.0, size_.width, size_.height);
    result.best_score = score[best_idx];

    return result;
}


cv::Mat NanoTracker::get_subwindow(    
    const cv::Mat& im,
    std::vector<float> pos,
    int model_sz,
    int original_sz,
    const cv::Scalar& avg_chans) {
    
    std::vector<float> center_pos = pos;
    if (center_pos.size() == 1) {
        center_pos.push_back(pos[0]);
    }
    
    int sz = original_sz;
    int r = im.rows;
    int c = im.cols;
    int k = im.channels();
    float context_c = (original_sz + 1) / 2.0f;

    int r_c = std::max(r, c);
    if(sz>r_c)
    {
        sz=r_c;
        context_c = (sz + 1) / 2.0f;
    }

    // 计算上下文边界
    float context_xmin = std::floor(center_pos[0] - context_c + 0.5f);
    float context_xmax = context_xmin + sz - 1;
    float context_ymin = std::floor(center_pos[1] - context_c + 0.5f);
    float context_ymax = context_ymin + sz - 1;

    // 计算填充大小
    int left_pad = std::max(0, static_cast<int>(-context_xmin));
    int top_pad = std::max(0, static_cast<int>(-context_ymin));
    int right_pad = std::max(0, static_cast<int>(context_xmax - c + 1));
    int bottom_pad = std::max(0, static_cast<int>(context_ymax - r + 1));

    // 调整上下文边界
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    

    
    // 创建填充后的图像
    cv::Mat te_im;
    if (left_pad > 0 || right_pad > 0 || top_pad > 0 || bottom_pad > 0) 
    {        
        im.copyTo(padded_image(cv::Rect(left_pad, top_pad, c, r)));

        // 填充边界（逐通道填充）
        for (int channel = 0; channel < k; ++channel) {
            if (top_pad > 0) {
                padded_image(cv::Rect(left_pad, 0, c, top_pad)).setTo(cv::Scalar(avg_chans[channel]));
            }
            if (bottom_pad > 0) {
                padded_image(cv::Rect(left_pad, r + top_pad, c, bottom_pad)).setTo(cv::Scalar(avg_chans[channel]));
            }
            if (left_pad > 0) 
            {
                padded_image(cv::Rect(0, top_pad, left_pad, r)).setTo(cv::Scalar(avg_chans[channel]));
            }

            if (right_pad > 0) 
            {             
                padded_image(cv::Rect(c + left_pad, top_pad, right_pad, r)).setTo(cv::Scalar(avg_chans[channel]));
            }
        }
        te_im = padded_image;
    } else {
        te_im = im.clone();
    }


    // 提取子窗口
    cv::Mat im_patch = te_im(cv::Rect(
        static_cast<int>(context_xmin),
        static_cast<int>(context_ymin),
        static_cast<int>(context_xmax - context_xmin + 1),
        static_cast<int>(context_ymax - context_ymin + 1)
    ));
        
    // 调整大小
    if (model_sz != original_sz) {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));        
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(im_patch, blob, 1.0, cv::Size(model_sz, model_sz), cv::Scalar(0, 0, 0), false, false);
            
    return blob;
}




// 定义 generate_points 函数
std::vector<std::vector<float>> NanoTracker::generate_points(int stride, int size) {
    // 计算起始点 ori
    float ori = - (size / 2.0f) * stride;

    // 初始化网格点
    std::vector<float> x_coords(size);
    std::vector<float> y_coords(size);

    // 填充 x 和 y 坐标
    for (int i = 0; i < size; ++i) {
        x_coords[i] = ori + stride * i;
        y_coords[i] = ori + stride * i;
    }

    // 创建 points 矩阵，大小为 size * size 行，每行包含两个值 (x, y)
    std::vector<std::vector<float>> points(size * size, std::vector<float>(2));

    // 填充 points
    int index = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            points[index][0] = x_coords[j]; // x 坐标
            points[index][1] = y_coords[i]; // y 坐标
            ++index;
        }
    }

    return points;
}

std::vector<std::vector<float>> NanoTracker::corner2center(const std::vector<std::vector<float>>& corner) {
    // 输入 corner 的形状为 (4, N)
    size_t num_points = corner[0].size();

    // 初始化结果矩阵 (4, N)，存储 (x, y, w, h)
    std::vector<std::vector<float>> result(4, std::vector<float>(num_points));

    for (size_t i = 0; i < num_points; ++i) {
        float x1 = corner[0][i];
        float y1 = corner[1][i];
        float x2 = corner[2][i];
        float y2 = corner[3][i];

        // 计算中心点 (x, y) 和宽高 (w, h)
        result[0][i] = (x1 + x2) * 0.5f; // x
        result[1][i] = (y1 + y2) * 0.5f; // y
        result[2][i] = x2 - x1;          // w
        result[3][i] = y2 - y1;          // h
    }

    return result;
}

std::vector<float> NanoTracker::hanning(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}


std::vector<float> NanoTracker::outer(const std::vector<float>& a, const std::vector<float>& b) {
    int rows = a.size();
    int cols = b.size();
    std::vector<float> result(rows * cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] = a[i] * b[j];
        }
    }

    return result;
}




Ort::Value NanoTracker::create_input_tensor(const cv::Mat& mat, const std::vector<int64_t>& shape) {
    // 确保输入是 float 类型的单通道或三通道图像
    CV_Assert(mat.type() == CV_32FC1 || mat.type() == CV_32FC3);

    // 创建 CPU 内存信息
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 获取数据指针和元素数量
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }

    // 确保 mat 中的数据数量与 shape 匹配
    CV_Assert(mat.total() * mat.channels() == num_elements);

    // 创建 Tensor
    return Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(mat.data),
        num_elements,
        shape.data(),
        shape.size()
    );
}



template<typename T>
std::pair<T*, std::vector<int64_t>> NanoTracker::get_tensor_data(Ort::Value& tensor) {
    T* data = tensor.GetTensorMutableData<T>();
    std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    return {data, shape};
}
    