
#ifndef _CLASS_HARRIS_HPP_
#define _CLASS_HARRIS_HPP_


#include <iostream>
#include <opencv2/opencv.hpp>

class Harris
{
public:
    Harris(/* args */)
    {}
    ~Harris()
    {}

    void detect(const cv::Mat &img_,std::vector<cv::Point> &corners_)
    {
        get_gradient(img_);
        cv::Mat kernel = gen_gaussion_kernel();
        Ixx = filter_float(Ixx,kernel);
        Iyy = filter_float(Iyy,kernel);
        Ixy = filter_float(Ixy,kernel);
        cv::Mat res = score_img(img_);
        float thresh =  34*abs(cv::mean(res)[0]);
        get_corners(res,thresh,corners_);
    }
private:

    void get_corners(const cv::Mat &res_,const float thresh_ ,std::vector<cv::Point> &corners_)
    {
        corners_.clear();
       for (int r = 0; r < res_.rows; r++)
        {
            for(int c = 0; c < res_.cols; c++)
            { 
                if (res_.at<float>(r,c)>thresh_)
                {
                    corners_.emplace_back(c,r);
                }
            }
        }
    }

    cv::Mat score_img(const cv::Mat &in_)
    {
        cv::Mat res = cv::Mat::zeros(in_.size(),CV_32FC1);
        for (int r = 0; r < in_.rows; r++)
        {
            for(int c = 0; c < in_.cols; c++)
            {
                cv::Mat M = (cv::Mat_<float>(2,2)<<Ixx.at<float>(r,c),Ixy.at<float>(r,c),Ixy.at<float>(r,c),Iyy.at<float>(r,c));
                float score = cv::determinant(M) - 0.05*cv::trace(M)[0]*cv::trace(M)[0];
                res.at<float>(r,c) = score;
            }
        }
        return res;
    }
    void get_gradient(const cv::Mat &in_)
    {
        cv::Mat sobelx = (cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
        cv::Mat sobely = (cv::Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
        cv::Mat Ix= filter_uchar(in_,sobelx);
        cv::Mat Iy= filter_uchar(in_,sobely);
        Ixx = Ix.mul(Ix);
        Iyy = Iy.mul(Iy);
        Ixy = Ix.mul(Iy);
    }

     cv::Mat filter_uchar(const cv::Mat &in_,const cv::Mat &kernel_)
        {
            cv::Mat res = cv::Mat::ones(in_.size(),CV_32FC1);
            int kernel_size = kernel_.rows;
            for (size_t r = 0; r < in_.rows; ++r)
            {
                for(size_t c =0 ;c<in_.cols;++c)
                {
                    float v =0;
                    //int cnt = 0;
                   for (int rk = -kernel_size/2; rk < kernel_size/2+1; ++rk)
                    {
                        for(int ck = -kernel_size/2;ck<kernel_size/2+1;++ck)
                        {
                            if((r+rk)<0||(r+rk)>=in_.rows||(c+ck)<0||(c+ck)>=in_.cols)continue;
                            float kv = kernel_.at<float>(rk+kernel_size/2,ck+kernel_size/2);
                            v+=float(in_.at<uchar>(r+rk,c+ck))*kv;
                        }
                    }
                    res.at<float>(r,c) =v;
                }
            }
            return res;
        }

    cv::Mat filter_float(const cv::Mat &in_,const cv::Mat &kernel_)
    {
        cv::Mat res = cv::Mat::zeros(in_.size(),CV_32FC1);
        int kernel_size = kernel_.rows;
        for (size_t r = 0; r < in_.rows; ++r)
        {
            for(size_t c =0 ;c<in_.cols;++c)
            {
                float v =0;
                //int cnt = 0;
               for (int rk = -kernel_size/2; rk < kernel_size/2+1; ++rk)
                {
                    for(int ck = -kernel_size/2;ck<kernel_size/2+1;++ck)
                    {
                        if((r+rk)<0||(r+rk)>=in_.rows||(c+ck)<0||(c+ck)>=in_.cols)continue;
                        float kv = kernel_.at<float>(rk+kernel_size/2,ck+kernel_size/2);
                        v+=in_.at<float>(r+rk,c+ck)*kv;
                    }
                }
                res.at<float>(r,c) =v;
            }
        }
        return res;
    }

    float gaussion(int x,int y,float theta)
    {
        return exp(-(x*x+y*y)/(2*theta*theta))/(2*theta*theta);
    }

    cv::Mat gen_gaussion_kernel()
    {
        cv::Mat kernel = cv::Mat::zeros(cv::Size(_kernel_size,_kernel_size),CV_32FC1);
        for (int r = -_kernel_size/2; r < _kernel_size/2+1; ++r)
        {
            for(int c = -_kernel_size/2;c<_kernel_size/2+1;++c)
            {
                //std::cout<<"r:"<<r<<" c:"<<c<<std::endl;
                float v = gaussion(c,r,_delta);
               // std::cout<<v<<std::endl;
                kernel.at<float>(r+_kernel_size/2,c+_kernel_size/2) = v;
            }
        }
        float sum = cv::sum(kernel)[0];
      //  std::cout<<kernel<<std::endl;
      //  std::cout<<sum<<std::endl;
        return kernel/sum;
            
    }

private:
    cv::Mat Ixx,Iyy,Ixy;
    int _kernel_size =5;
    float _delta = 1.4;
};

#endif // !1_CLASS_HARRIS_HPP_
