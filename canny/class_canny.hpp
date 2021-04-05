#ifndef CLASS_CANNY_HPP_
#define CLASS_CANNY_HPP_
#include <opencv2/opencv.hpp>
#include <math.h>
namespace EZ
{
    class Canny
    {
    private:
        /* data */
    public:
        Canny(/* args */)
        {

        }
        void set_params(const int kernel_size = 5,float delta_=1.4)
        {
            _kernel_size = kernel_size;
            _delta = delta_;
        }

        void run(const cv::Mat &img_,cv::Mat &res_)
        {
            _img = img_.clone();
            gaussion_blur(res_);
            cv::imshow("blur",res_);
            cv::imwrite("../res/blur.jpg",res_);
            cv::Mat gradient,direction;
            calculate_gradient(res_,gradient,direction);
            cv::Mat gradient_num = nms(gradient,direction);
            cv::imshow("nms",gradient_num);
            cv::imwrite("../res/nms.jpg",gradient_num);
            cv::Mat mat_thresh = double_thresh(gradient_num,0.05,0.09);
            cv::imshow("thresh",mat_thresh);
            cv::imwrite("../res/thresh.jpg",mat_thresh);
            cv::Mat mat_canny = track_hysteresis(mat_thresh);
            cv::imshow("canny",mat_canny);
            cv::imwrite("../res/canny.jpg",mat_canny);
            cv::waitKey();
        }

        ~Canny()
        {

        }
    private:

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
        cv::Mat filter(const cv::Mat &in_,const cv::Mat &kernel_)
        {
            cv::Mat res = cv::Mat::ones(in_.size(),CV_8UC1);
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
                            v+=in_.at<uchar>(r+rk,c+ck)*kv;
                        }
                    }
                    res.at<uchar>(r,c) =(uchar)v;
                }
            }
            return res;
        }

        cv::Mat filter_float(const cv::Mat &in_,const cv::Mat &kernel_)
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
                            v+=in_.at<char>(r+rk,c+ck)*kv;
                        }
                    }
                    res.at<float>(r,c) =v;
                }
            }
            return res;
        }
        void gaussion_blur(cv::Mat &res_)
        {
           // res_ = cv::Mat::zeros(_img.size(),CV_8UC1);
            cv::Mat kernel = gen_gaussion_kernel();
            std::cout<<kernel<<std::endl;
            res_ = filter(_img,kernel);
           // std::cout<<res_<<std::endl;
        }

        void calculate_gradient(cv::Mat &res_,cv::Mat &intensity_,cv::Mat &direction_)
        {
            cv::Mat sobelx = (cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
            cv::Mat sobely = (cv::Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
            cv::Mat gradientX = filter_float(res_,sobelx);
           // std::cout<<gradientX<<std::endl;
           // std::cout<<gradientX<<std::endl;
            cv::Mat gradientY = filter_float(res_,sobely);
            cv::imwrite("../res/gradientX.jpg",gradientX);
            cv::imwrite("../res/gradientY.jpg",gradientY);
            res_ = gradientX.mul(gradientX)+gradientY.mul(gradientY);
            double min=0,max=0;
            for (int r  = 0; r < res_.rows; r++)
            {
                for (int c = 0; c < res_.cols; c++)
                {
                    res_.at<float>(r,c) = sqrtf(res_.at<float>(r,c));
                }
            }
            
            cv::minMaxIdx(res_,&min,&max);
          //  std::cout<<max<<std::endl;
            res_ = res_/max*255;
           // std::cout<<res_<<std::endl;
            res_.convertTo(intensity_,CV_8UC1);

            for (int r  = 0; r < res_.rows; r++)
            {
                for (int c = 0; c < res_.cols; c++)
                {
                    res_.at<float>(r,c) = sqrtf(res_.at<float>(r,c));
                }
            }
            direction_ = cv::Mat::zeros(res_.size(),CV_32FC1);
            for (int r  = 0; r < res_.rows; r++)
            {
                for (int c = 0; c < res_.cols; c++)
                {
                   // res_.at<float>(r,c) = sqrtf(res_.at<float>(r,c));
                    direction_.at<float>(r,c) = atan2f(gradientY.at<float>(r,c),gradientX.at<float>(r,c));
                }
            }
         //   std::cout<<direction_<<std::endl;
          //  cv::imshow("Gx",gradientX);
          //  cv::imshow("Gy",gradientY);
            cv::imwrite("../res/intensity.jpg",intensity_);
            cv::imshow("G",intensity_);

        }

        cv::Mat nms(const cv::Mat& gradient_,const cv::Mat &direction_)
        {
            cv::Mat gradient_nms = cv::Mat::zeros(gradient_.size(),CV_8UC1);
            for (int r = 1; r < gradient_.rows-1; r++)
            {
                for (int c = 1; c < gradient_.cols-1; c++)
                {
                   float dir = direction_.at<float>(r,c)*180/CV_PI;
                   dir = (dir<0)?(dir+180):dir;
                   uchar p=255,q=255;
                   if ((dir>=0&&dir<22.5)||(dir>=157.5||dir<=180))
                   {
                       p = gradient_.at<uchar>(r,c-1);
                       q = gradient_.at<uchar>(r,c+1);
                   }
                   else if(dir>=22.5&&dir<67.5)
                   {
                       p = gradient_.at<uchar>(r+1,c-1);
                       q = gradient_.at<uchar>(r-1,c+1);

                   }
                   else if (dir>=67.5&&dir<112.5)
                   {
                       p = gradient_.at<uchar>(r-1,c);
                       q = gradient_.at<uchar>(r+1,c);
                       
                   }
                   else if (dir>=112.5&&dir<157.5)
                   {
                       p = gradient_.at<uchar>(r-1,c-1);
                       q = gradient_.at<uchar>(r+1,c+1);
                       
                   }

                   if (gradient_.at<uchar>(r,c)>=p &&gradient_.at<uchar>(r,c)>=q)
                   {
                       gradient_nms.at<uchar>(r,c) = gradient_.at<uchar>(r,c);
                   }
                   else
                   {
                       gradient_nms.at<uchar>(r,c) = 0;
                   }
                   
                   
                     
                }
            }
            return gradient_nms;
            
        }

        cv::Mat double_thresh(cv::Mat &gradient_,float low_ratio_,float high_ratio_)
        {
            cv::Mat mat_thresh = cv::Mat::zeros(gradient_.size(),CV_8UC1);
            double min,max;
            cv::minMaxIdx(gradient_,&min,&max);
            float high_thresh = max*high_ratio_;
            float low_thresh = high_thresh*low_ratio_;
            for (int r = 1; r < gradient_.rows-1; r++)
            {
                for (int c = 1; c < gradient_.cols-1; c++)
                {

                   if(gradient_.at<uchar>(r,c)>high_thresh)
                   {
                       mat_thresh.at<uchar>(r,c) = 255;
                   }
                   else if (gradient_.at<uchar>(r,c)>low_thresh)
                   {
                       mat_thresh.at<uchar>(r,c) = 25;
                   }
                }
            }
            return mat_thresh;
        }

        cv::Mat track_hysteresis(const cv::Mat &mat_thresh_)
        {
            cv::Mat mat_canny = cv::Mat::zeros(mat_thresh_.size(),CV_8UC1);

            for (int r = 1; r < mat_thresh_.rows-1; r++)
            {
                for (int c = 1; c < mat_thresh_.cols-1; c++)
                {
                    if (mat_thresh_.at<uchar>(r,c)==25)
                    {
                    
                        if (mat_thresh_.at<uchar>(r-1,c-1)>=255  ||
                            mat_thresh_.at<uchar>(r-1,c)>=255  ||
                            mat_thresh_.at<uchar>(r-1,c+1)>=255  ||
                            mat_thresh_.at<uchar>(r,c-1)>=255  ||
                            mat_thresh_.at<uchar>(r,c)>=255  ||
                            mat_thresh_.at<uchar>(r,c+1)>=255  ||
                            mat_thresh_.at<uchar>(r+1,c-1)>=255  ||
                            mat_thresh_.at<uchar>(r+1,c)>=255  ||
                            mat_thresh_.at<uchar>(r+1,c+1)>=255 )
                        {
                            mat_canny.at<uchar>(r,c) = 255;
                        }
                    }
                    else if (mat_thresh_.at<uchar>(r,c)==255)
                    {
                       mat_canny.at<uchar>(r,c) = 255; 
                    }
                }
            }
            return mat_canny;
        }
    private :
        cv::Mat _img;
        int _kernel_size;
        float _delta;
    };
};

#endif // !1CLASS_CANNY_HPP_

