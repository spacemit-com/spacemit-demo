#include"mobileclip.hpp"
#include"yoloe.hpp"
#include <iostream>
#include<sstream>
#include <iomanip>

int main(int argc,char** argv) 
{
    std::string mobileclip_text_modelPath;
    std::string yoloe_modelPath;
    std::string imagePath;
    std::vector<std::string> test_strings; 
    test_strings.clear();
    for(int i=1; i<argc;++i)
    {
      std::string arg=argv[i];
      if(arg=="--mobileclip_model"&&i+1<argc)
      {
        mobileclip_text_modelPath=argv[++i];
      }
      else if(arg=="--yoloe_model"&&i+1<argc)
      {
        yoloe_modelPath=argv[++i];
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
    
    
    MOBILECLIP mobileclip_instance(mobileclip_text_modelPath);
    mobileclip_instance.inference_clip_model(test_strings,test_strings.size());
    YOLOE yoloe_instance(yoloe_modelPath);
    for(int i=0;i<test_strings.size();i++)
    {
      yoloe_instance.labels.push_back(test_strings[i]);
    }
    cv::Mat image = cv::imread(imagePath);
    yoloe_instance.init_yoloe_model(mobileclip_instance.clip_data_,image,0.25,0.45);
    cv::Mat result_img=yoloe_instance.inference_yoloe_model(image);
    cv::imwrite("result.jpg", result_img);
    printf("Result image saved as result.jpg\n");
  
    return 0;
}
