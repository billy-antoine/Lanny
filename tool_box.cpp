#include "tool_box.h"

using namespace cv;
using namespace std;

void getCannyMask(cv::Mat &inputImg, cv::Mat &dispMap, cv::Mat &outputImg){

    cv::Mat contours, grayscale, mask;

    inputImg.convertTo(grayscale, CV_8U);

    cv::blur(grayscale, grayscale, cv::Size(3,3));
    cv::Canny(grayscale, contours, 100, 200);

    contours.convertTo(mask, CV_8U);
    cv::threshold(mask, mask, 100, 255, CV_THRESH_BINARY);
    //cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

    dispMap.copyTo(outputImg, mask);

}

