#ifndef STEREO_CALIB
#define STEREO_CALIB

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/core.hpp>
#include "dirent.h"
#include <errno.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

void loadIntrinsics(const std::string path, cv::Mat &M1, cv::Mat &M2, cv::Mat &D1, cv::Mat &D2);
void loadExtrinsics(const std::string path, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &R, cv::Mat &T, cv::Mat &Q);
void loadRoIs(const std::string path, cv::Rect &roi1, cv::Rect &roi2);
std::vector<std::string> getCalibrationData(int num_cam1, int num_cam2, const char* imagelistfn);
void StereoCalib(const std::vector<std::string>& imagelist, std::string outputFolder, cv::Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated = true, bool showRectified = true);
int getdir (std::string dir, std::vector<std::string> &files);

#endif
