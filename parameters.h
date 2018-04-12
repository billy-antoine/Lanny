#ifndef PARAMETERS
#define PARAMETERS

#include <opencv2/highgui.hpp>

//StereoMotion
int densification_ratio = 25;
bool use_icp = false;

bool multi_threading = true;
int ROI_size = 80;
double resize_factor = 1;

// Calibration parameters
bool showRectified = true;
bool displayCorner = true;
bool useCalibrated = true;
bool calibration = false;

bool writeImage = true;
bool displayImages = true;
bool display3D = false;

cv::Size boardSize(9,6);
float squareSize = 3.0f;
//const char* imagelistfn = "Data/Calibration_images/test3/";
//const string inputDir = "/home/doctorant/Vidéos/Stereo/Winster/kitti/";

// SGBM parameters
int s_SADWindowSize = 3;
int s_numberOfDisparities = 5;
int s_preFilterCap = 48;
int s_minDisparity = 64;
int s_uniquenessRatio =2;
int s_speckleWindowSize = 150;

int s_speckleRange = 32;
int s_disp12MaxDiff = 11;

// Post-processing parameters
int s_write3D = 0;
int s_closing = 0;
int s_inpainting = 0;
int s_scale = 40; // In percentage
int s_mblur = 3;
int s_color = 1;

// 3D parameters
int maxDepth = 120;
int ratioCrop3D = 10; // Percent of croped borders for 3D point cloud
int cloudNumber = 0;

//Camera resolution
int PROP_FRAME_WIDTH = 1280;
int PROP_FRAME_HEIGHT = 960;

float cx=6.071928e+02;
float cy=1.852157e+02;
float focal=7.188560e+02;
float baseline_length=540.0;

cv::Mat Q = (cv::Mat_<float>(4,4) <<   1., 0., 0., -cx,
                            0., 1., 0., -cy,
                            0., 0., 0., focal,
                            0., 0., -1., 0);


#endif
