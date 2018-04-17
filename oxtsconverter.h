//#pragma once

//#include <opencv2/opencv.hpp>
//#include "opencv2/core.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/xfeatures2d.hpp"

//#include <vector>
//#include <string>
//#include <algorithm>
//#include <iostream>
//#include <iterator>
//#include <stdio.h>
//#include <stdlib.h>
//#include <cctype>
//#include <math.h>
//#include <Eigen/Dense>
//#include <Eigen/Geometry>

//#include <fstream>

//struct oxts {
//    double x;
//    double y;
//    double z;

//    double lat;
//    double lon;
//    double alt;

//    double rx;
//    double ry;
//    double rz;

//    double roll;
//    double pitch;
//    double yaw;
//};
//typedef struct oxts Oxts;

//void latlonMercator(Oxts& src, double scale){
//    double er = 6378137;
//    mx = scale * src.lon * PI * er / 180;
//    my = scale * er * log( tan((90+ src.lat) * PI / 360) );
//    src.x = mx;
//    src.y = my;
//}

//double latToScale(double lat){
//   return cos(lat * PI / 180.0);
//}

//// base -> repere
//// out remplit de base avec lat lon alt yaw pitch roll -> reste
//void OXTSConverter(Oxts& base, Oxts& out ){
//    double scale = latToScale(base.lat);

//    latlonToMercator(out, scale);
//    out.z = out.alt;

//    double rx = out.roll;
//    double ry = out.pitch;
//    double rz = out.yaw;

//    cv::Mat Rx = cv::Mat_<float>(4,4) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx);
//    cv::Mat Ry = cv::Mat_<float>(4,4) << cos(ry), 0, sin(ry), 0, 1, 0, -sin(ry), 0, cos(ry);
//    cv::Mat Rz = cv::Mat_<float>(4,4) << 1, 0, 0, 0, cos(rx), -sin(rx), 0, sin(rx), cos(rx);

//    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];
//    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1];
//    R  = Rz*Ry*Rx;

//    if isempty(Tr_0_inv)
//      Tr_0_inv = inv([R t;0 0 0 1]);
//    end

//    pose{i} = Tr_0_inv*[R t;0 0 0 1];


//}
