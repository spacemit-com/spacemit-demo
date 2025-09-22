#include"clip.hpp"
CLIP::CLIP(const std::string& text_model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "CLIP"),
      session1_(nullptr)
{  
    Ort::SessionOptions  session_options_1;
    session_options_1.SetIntraOpNumThreads(4);
    //SessionOptionsSpaceMITEnvInit(session_options_1);
    session1_ = std::make_unique<Ort::Session>(env_, text_model_path.c_str(),session_options_1);
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
    
    tokenizer=new CLIPTokenizer();
    sequence_len=77;
}
CLIP::~CLIP()
{
    delete tokenizer;
}

Ort::Value CLIP::create_input_tensor(const std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(data.data()), data.size(), shape.data(), shape.size());
}

template<typename T>
std::pair<T*, std::vector<int64_t>> CLIP::get_tensor_data(Ort::Value& tensor) 
{
    T* data = tensor.GetTensorMutableData<T>();
    std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    return {data, shape};
}
    
void CLIP::inference_clip_model(std::vector<std::string> input_texts,int text_num)
{
    clip_data_.clear();
    std::vector<int64_t> input_shape = {1, sequence_len};
    for (int i = 0; i < text_num; i++)
    {
        std::vector<int32_t> input=tokenizer->tokenize(input_texts[i])[0];
        std::vector<int64_t> token;
        token.reserve(input.size());
        for (auto x : input) {
            token.push_back(static_cast<int64_t>(x));
        }
        Ort::Value input_tensor = create_input_tensor(token, input_shape);
        std::vector<Ort::Value> outputs = session1_->Run(Ort::RunOptions{nullptr}, input1_names_.data(), &input_tensor, 1, output1_names_.data(), output1_names_.size());
        auto [data, shape] = get_tensor_data<float>(outputs[0]);
        clip_data.assign(data, data + shape[0] * shape[1]); 
        clip_data_.push_back(std::move(clip_data));
    }
}

