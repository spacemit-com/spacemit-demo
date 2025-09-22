#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include<sstream>
#include <iomanip>
#include"mobilesam.h"
int main(int argc, char **argv)
{
    if (argc != 6)
    {
        printf("%s <encoder_model_path> <decoder_model_path> <image_path> <point_coords_path> <point_labels_path>\n", argv[0]);
        return -1;
    }
    const char *encoder_model_path = argv[1];
    const char *decoder_model_path = argv[2];
    const char *img_path = argv[3];
    const char *point_coords_path = argv[4];
    const char *point_labels_path = argv[5];

    Mobile_Sam  mobile_sam_instance(encoder_model_path,decoder_model_path);
  
    cv::Mat image = cv::imread(img_path);
    mobile_sam_instance.init_Mobile_Sam(point_coords_path,point_labels_path,image);
    
    
    mobile_sam_instance.inference_mobilesam_model(image);
    mobile_sam_instance.draw_mask(image,mobile_sam_instance.res_mask);
    cv::imwrite("result.jpg", image);
    printf("Result image saved as result.jpg\n");


}
