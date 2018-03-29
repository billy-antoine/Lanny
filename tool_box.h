#ifndef TOOL_BOX
#define TOOL_BOX

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <cctype>

#include <fstream>


void getCannyMask(cv::Mat &inputImg, cv::Mat &dispMap, cv::Mat &outputImg);


void getFeaturesMask(cv::Mat &img_0, cv::Mat &img_1, cv::Mat &dispMap_0, cv::Mat &dispMap_1, cv::Mat &featureMap_0, cv::Mat &featureMap_1);
#endif
