
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    for (;;)
    {
        Mat img = imread("./lena.jpg", IMREAD_GRAYSCALE);
        if (img.empty())
        {
            printf("Cannot read image file: %s\n");
            return -1;
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        int M = getOptimalDFTSize(img.rows);
        int N = getOptimalDFTSize(img.cols);
        Mat padded;
        copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

        Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
        Mat complexImg;
        merge(planes, 2, complexImg);

        dft(complexImg, complexImg);

        // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
        split(complexImg, planes);
        magnitude(planes[0], planes[1], planes[0]);
        Mat mag = planes[0];
        mag += Scalar::all(1);
        log(mag, mag);

        // crop the spectrum, if it has an odd number of rows or columns
        mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

        int cx = mag.cols / 2;
        int cy = mag.rows / 2;

        // rearrange the quadrants of Fourier image
        // so that the origin is at the image center
        Mat tmp;
        Mat q0(mag, Rect(0, 0, cx, cy));
        Mat q1(mag, Rect(cx, 0, cx, cy));
        Mat q2(mag, Rect(0, cy, cx, cy));
        Mat q3(mag, Rect(cx, cy, cx, cy));

        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        normalize(mag, mag, 0, 1, NORM_MINMAX);
        auto endTime = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        std::cout << "total time:" << totalTime << std::endl;
        imshow("spectrum magnitude", mag);
        waitKey(10);
    }
    
    return 0;

}