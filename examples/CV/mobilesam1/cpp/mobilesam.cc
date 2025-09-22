#include"mobilesam.h"
#include <cmath>
#include <chrono>
#include <iostream>
Mobile_Sam::Mobile_Sam(const std::string& onnx_encoder_path,
                         const std::string& onnx_decoder_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Mobile_Sam"),
      session1_(nullptr),
      session2_(nullptr){

    Ort::SessionOptions session_options_1;
    session_options_1.SetIntraOpNumThreads(4);
    //SessionOptionsSpaceMITEnvInit(session_options_1);

    Ort::SessionOptions session_options_2;       
    session_options_2.SetIntraOpNumThreads(4);
    //SessionOptionsSpaceMITEnvInit(session_options_2);

    session1_ = std::make_unique<Ort::Session>(env_, onnx_encoder_path.c_str(),session_options_1);
    session2_ = std::make_unique<Ort::Session>(env_, onnx_decoder_path.c_str(), session_options_2);

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
    ori_point_coords=NULL;
    point_labels=NULL;
    mask_input=NULL;
    res_mask=NULL;
    
}

Mobile_Sam::~Mobile_Sam()
{
    if(NULL!=ori_point_coords)
    {
         free(ori_point_coords);
    }
    if (point_labels != NULL)
    {
        free(point_labels);
    }
    if (mask_input != NULL)
    {
        free(mask_input);
    }
    if(res_mask!=NULL)
    {
         free(res_mask);
    }

}

int Mobile_Sam::max(int a,int b)
{
  return a>b? a:b;
}

void Mobile_Sam::Preprocess(const cv::Mat& image, cv::Mat& blob_image)
{
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::Scalar std(58.395, 57.12, 57.375);  
    int orig_width = image.cols;
    int orig_height = image.rows;
    float scale = IMG_SIZE * 1.0 / max(orig_height, orig_width);
    int scaled_width = static_cast<int>(orig_width * scale);
    int scaled_height = static_cast<int>(orig_height * scale);
    int offset_width = (IMG_SIZE - scaled_width) / 2;
    int offset_height = (IMG_SIZE - scaled_height) / 2;
    cv::Mat pad_image=cv::Mat(IMG_SIZE, IMG_SIZE, image.type(), cv::Scalar(0, 0, 0));

    letterbox.scaled_width = scaled_width;
    letterbox.scaled_height = scaled_height;
    letterbox.offset_width = offset_width;
    letterbox.offset_height = offset_height; 
    letterbox.scale=scale;

    for (int i = 0; i < coords_nums*2; i+=2)
    {
        ori_point_coords[i] = ori_point_coords[i] * letterbox.scale;
        ori_point_coords[i + 1] = ori_point_coords[i + 1] * letterbox.scale;
        //ori_point_coords[i] = ori_point_coords[i] * letterbox.scale+letterbox.offset_width;
        //ori_point_coords[i + 1] = ori_point_coords[i + 1] * letterbox.scale+ letterbox.offset_height;
    }
    
    cv::Mat canvas=pad_image(cv::Rect(0, 0, letterbox.scaled_width, letterbox.scaled_height));
    //cv::Mat canvas=pad_image(cv::Rect(letterbox.offset_width, letterbox.offset_height, letterbox.scaled_width, letterbox.scaled_height));
    cv::resize(image, canvas, cv::Size(letterbox.scaled_width, letterbox.scaled_height), 0, 0, cv::INTER_LINEAR);
    blob_image = cv::dnn::blobFromImage(pad_image, 1.0, cv::Size(pad_image.cols, pad_image.rows), mean, true, false, CV_32F);
    int channels = blob_image.size[1];
    int height = blob_image.size[2];
    int width = blob_image.size[3];
    for (int c = 0; c < channels; ++c) {
        cv::Mat channel(height, width, CV_32F, blob_image.ptr(0, c));
        channel /= std[c];
    }
    std::cout<<"Preprocess_end!!!"<<std::endl;

}

void Mobile_Sam::init_Mobile_Sam(const char* point_coords_path, const char* point_labels_path,const cv::Mat& img)
{
    ori_point_coords = read_coords_from_file(point_coords_path, &coords_nums);
    point_labels = read_coords_from_file(point_labels_path, &labels_nums);
    mask_input=(float*)malloc((IMG_SIZE/4)*(IMG_SIZE/4)*sizeof(float));
    memset(mask_input, 0, (IMG_SIZE/4)*(IMG_SIZE/4)* sizeof(float));
    has_mask_input=0;
    res_mask = (uint8_t*)malloc(img.rows * img.cols * sizeof(uint8_t));
}

int Mobile_Sam::count_lines(FILE* file)
{
    int count = 0;
    char ch;
    while(!feof(file))
    {
        ch = fgetc(file);
        if(ch == '\n')
        {
            count++;
        }
    }
    //count += 1;

    rewind(file);
    return count;
}


float* Mobile_Sam::read_coords_from_file(const char* filename, int* line_count)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return NULL;
    }

    int num_lines = count_lines(file);
    //printf("num_lines=%d\n", num_lines);
    float* coords = (float*)malloc(num_lines * 2 * sizeof(float));
    memset(coords, 0, num_lines * 2 * sizeof(float));

    char buffer[MAX_TEXT_LINE_LENGTH];
    int index = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        buffer[strcspn(buffer, "\n")] = ' ';  // 移除换行符

        char* coord = strtok(buffer, " ");
        while (coord != NULL) 
        {
            coords[index++] = atof(coord);
            coord = strtok(NULL, " ");
        }
    }

    fclose(file);
    *line_count = num_lines;
    return coords;
}

void Mobile_Sam::inference_mobilesam_model(const cv::Mat& img)
{
    cv::Mat inputTensor;
    Preprocess(img, inputTensor);
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(5);
    std::vector<int64_t> input_shape = {1, 3, IMG_SIZE, IMG_SIZE};
    Ort::Value input_tensor = create_input_tensor(inputTensor, input_shape);
    
    
    auto outputs1 = session1_->Run(Ort::RunOptions{nullptr}, input1_names_.data(), &input_tensor, 1, output1_names_.data(), output1_names_.size());
    
    
    //std::vector<int64_t> input1_shape = {1, 3, IMG_SIZE/16, IMG_SIZE/16};
    input_tensors.push_back(std::move(outputs1[0]));

    std::vector<int64_t> input2_shape = {1, coords_nums, 2};
    Ort::MemoryInfo memory_info2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value vpoint_coords = Ort::Value::CreateTensor<float>(memory_info2, ori_point_coords, coords_nums*2, input2_shape.data(), input2_shape.size());
    input_tensors.push_back(std::move(vpoint_coords));

    std::vector<int64_t> input3_shape = {1, labels_nums};
    Ort::MemoryInfo memory_info3 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value vpoint_labels = Ort::Value::CreateTensor<float>(memory_info3, point_labels, labels_nums, input3_shape.data(), input3_shape.size());
    input_tensors.push_back(std::move(vpoint_labels));

    std::vector<int64_t> input4_shape = {1, 1,IMG_SIZE/4,IMG_SIZE/4};
    Ort::MemoryInfo memory_info4 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value vmask_input = Ort::Value::CreateTensor<float>(memory_info4, mask_input, (IMG_SIZE/4)*(IMG_SIZE/4), input4_shape.data(), input4_shape.size());
    input_tensors.push_back(std::move(vmask_input));

    std::vector<int64_t> input5_shape = {1};
    Ort::MemoryInfo memory_info5 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value vhas_mask_input = Ort::Value::CreateTensor<float>(memory_info5, &has_mask_input, 1, input5_shape.data(), input5_shape.size());
    input_tensors.push_back(std::move(vhas_mask_input));

    std::vector<Ort::Value> outputs_final = session2_->Run(Ort::RunOptions{nullptr}, input2_names_.data(), input_tensors.data(), 5, output2_names_.data(), output2_names_.size());
    auto [score_data, score_shape] = get_tensor_data<float>(const_cast<Ort::Value&>(outputs_final[0]));
    auto [low_res_masks_data, low_res_masks_shape] = get_tensor_data<float>(const_cast<Ort::Value&>(outputs_final[1]));
    score_shape_=score_shape;
    masks_shape_=low_res_masks_shape;
    post_process(score_data,low_res_masks_data,img.rows,img.cols);
    
}
int Mobile_Sam::argmax(float* arr, int size)
{
    int index = 0;
    for (int i = 0; i < size; i++)
    {
        if (arr[i] > arr[index])
        {
            index = i;
        }
    }
    return index;
}

void Mobile_Sam::resize_by_opencv_fp(float *input_image, int input_height, int input_width, float *output_image, int target_height, int target_width)
{
    cv::Mat src_image(input_height, input_width, CV_32F, input_image);
    cv::Mat dst_image;
    cv::resize(src_image, dst_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
    memcpy(output_image, dst_image.data, target_width * target_height * sizeof(float));
}

void Mobile_Sam::slice_mask(float* ori_mask, float* slice_mask, int ori_width, int slice_height, int slice_width)
{
    for (int i = 0; i < slice_height; i++)
    {
        for (int j = 0; j < slice_width; j++)
        {
            slice_mask[i * slice_width + j] = ori_mask[i * ori_width + j];
        }
    }
}

void Mobile_Sam::crop_mask(float* ori_mask, int size, uint8_t* res_mask, float threshold)
{
    for (int i = 0; i < size; i++)
    {
        res_mask[i] = ori_mask[i] > threshold ? 1 : 0; 
    }
}

void Mobile_Sam::post_process(float* iou_predictions, float* low_res_masks, int ori_height, int ori_width)
{
     int masks_num = score_shape_[0]*score_shape_[1];
     int index = argmax(iou_predictions, masks_num);
     int low_res_masks_height = masks_shape_[2];
     int low_res_masks_width = masks_shape_[3];
     float* mask_img_size = (float*)malloc(IMG_SIZE * IMG_SIZE * sizeof(float));
     resize_by_opencv_fp(low_res_masks + index * low_res_masks_height * low_res_masks_width, low_res_masks_height, low_res_masks_width, mask_img_size, IMG_SIZE, IMG_SIZE);
     float* mask_new_shape = (float*)malloc( letterbox.scaled_width* letterbox.scaled_height*sizeof(float));
     float* mask_ori_img = (float*)malloc(ori_height * ori_width * sizeof(float));
     slice_mask(mask_img_size, mask_new_shape, IMG_SIZE, letterbox.scaled_height, letterbox.scaled_width);
     resize_by_opencv_fp(mask_new_shape, letterbox.scaled_height, letterbox.scaled_width, mask_ori_img, ori_height, ori_width);
     crop_mask(mask_ori_img, ori_height*ori_width, res_mask, MASK_THRESHOLD);
     if(mask_img_size!=NULL)
     {
         free(mask_img_size);
     }
     if(mask_new_shape!=NULL)
     {
         free(mask_new_shape);
     }
     if(mask_ori_img!=NULL)
     {
         free(mask_ori_img);
     }   
}

int Mobile_Sam::clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

void Mobile_Sam::draw_mask( cv::Mat& image, uint8_t* mask)
{
    int width = image.cols;
    int height = image.rows;

    unsigned char* ori_img =(unsigned char*) image.data;
    float alpha = 0.5f;
    int x1=0,y1=0,x2=0,y2=0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int pixel_offset = 3 * (i * width + j);
            if (mask[i * width + j] != 0)
            {
                ori_img[pixel_offset + 0] = (unsigned char)clamp(COLOR[0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255);
                ori_img[pixel_offset + 1] = (unsigned char)clamp(COLOR[1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255);
                ori_img[pixel_offset + 2] = (unsigned char)clamp(COLOR[2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255);
            }
        }
    }
    
    for(int i=0;i<labels_nums;i++)
    {
      if(point_labels[i]==2)
      {
        x1=int(ori_point_coords[i*2]/letterbox.scale);
        y1=int(ori_point_coords[i*2+1]/letterbox.scale);
      }
      else if(point_labels[i]==3)
      {
        x2=int(ori_point_coords[i*2]/letterbox.scale);
        y2=int(ori_point_coords[i*2+1]/letterbox.scale);
      }
    }
    if(x1!=0 || y1!=0 || x2!=0 || y2!=0 )
    {
      cv::rectangle(image,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,255,0),2);
    }

   
}

Ort::Value Mobile_Sam::create_input_tensor(const cv::Mat& mat, const std::vector<int64_t>& shape) 
{
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
std::pair<T*, std::vector<int64_t>> Mobile_Sam::get_tensor_data(Ort::Value& tensor) {
    T* data = tensor.GetTensorMutableData<T>();
    std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    return {data, shape};
}
    
