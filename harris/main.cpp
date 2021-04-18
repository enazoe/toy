#include "class_harris.hpp"

int main()
{
    Harris harris;
    
    cv::Mat img = cv::imread("../pattern.png",cv::IMREAD_GRAYSCALE);
    std::vector<cv::Point> corners;
    harris.detect(img,corners);
    cv::Mat show;
    cv::cvtColor(img,show,cv::COLOR_GRAY2BGR);
    std::cout<<"corner size:"<<corners.size()<<std::endl;
    for (const auto c:corners)
    {
        cv::circle(show,c,2,cv::Scalar(0,0,255),2);
    }
    cv::imshow("res",show);
    cv::imwrite("../result.jpg",show);
    cv::waitKey();
    return 0;
}