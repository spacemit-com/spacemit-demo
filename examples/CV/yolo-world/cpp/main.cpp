#include"clip.hpp"
#include"yolo_word.hpp"
#include <iostream>
#include<sstream>
#include <iomanip>

int main(int argc,char** argv) 
{
    std::string clip_text_modelPath;
    std::string yolo_world_modelPath;
    std::string imagePath;
    std::vector<std::string> test_strings; 
    test_strings.clear();
    for(int i=1; i<argc;++i)
    {
      std::string arg=argv[i];
      if(arg=="--clip_model"&&i+1<argc)
      {
        clip_text_modelPath=argv[++i];
      }
      else if(arg=="--yolo_model"&&i+1<argc)
      {
        yolo_world_modelPath=argv[++i];
      }
      else if(arg=="--image"&&i+1<argc)
      {
        imagePath=argv[++i];
      }
      else if(arg=="--text"&&i+1<argc)
      {
        std::stringstream ss(argv[++i]);
        std::string text;
        while (std::getline(ss,text,','))
        {
           test_strings.push_back(text);  
        }
      }
    }
    CLIP clip_instance(clip_text_modelPath);
    clip_instance.inference_clip_model(test_strings,test_strings.size());
    YOLO_WORLD yolo_world_instance(yolo_world_modelPath);
    for(int i=0;i<test_strings.size();i++)
    {
      yolo_world_instance.labels.push_back(test_strings[i]);
    }
    cv::Mat image = cv::imread(imagePath);
    yolo_world_instance.init_yolo_word_model(clip_instance.clip_data_,image,0.25,0.45);
    cv::Mat result_img=yolo_world_instance.inference_yolo_world_model(image);
    cv::imwrite("result.jpg", result_img);
    printf("Result image saved as result.jpg\n");
  
    return 0;
}
