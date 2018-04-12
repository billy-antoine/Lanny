#pragma once

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
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>


// PCL
#include <pcl/common/common_headers.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>


static const double earthRadius = 6378.1370; //km
static const float PI = 3.14159265;

template<typename T>
void pop_front(std::vector<T>& vec){
    assert(!vec.empty());
    vec.front() = std::move(vec.back());
    vec.pop_back();
}

void getCannyMask(cv::Mat &inputImg, cv::Mat &dispMap, cv::Mat &outputImg);

void llaToXyz(double lat, double lon, double alt, double &x, double &y, double &z);


void translate_rotate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc, double roll, double pitch, double yaw, Eigen::Vector3d trans);
void getFeaturesMask(cv::Mat &img_0, cv::Mat &img_1, cv::Mat &dispMap_0, cv::Mat &dispMap_1, cv::Mat &featureMap_0, cv::Mat &featureMap_1);

