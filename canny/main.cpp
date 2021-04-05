
#include <iostream>
#include "class_canny.hpp"

int main()
{
    std::cout<<"hello world!\n";
    EZ::Canny canny;
    canny.set_params();
    cv::Mat img = cv::imread("../lena.png",cv::IMREAD_GRAYSCALE);
    cv::namedWindow("lena",cv::WINDOW_NORMAL);
    cv::imshow("lena",img);
    cv::Mat res_canny;
    canny.run(img,res_canny);
    //cv::namedWindow("lena_canny",cv::WINDOW_NORMAL);
    //cv::imshow("lena_canny",res_canny);
    //cv::waitKey();
    return 0;
}