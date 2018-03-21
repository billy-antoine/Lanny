#ifndef TOOL_BOX
#define TOOL_BOX

#include <opencv2/opencv.hpp>

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

#endif
