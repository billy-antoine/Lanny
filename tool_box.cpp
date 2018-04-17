#include "tool_box.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

using namespace Eigen;


bool compare_DMatches(const cv::DMatch m1, const cv::DMatch m2){ return m1.distance < m2.distance; }

void llaToXyz(double lat, double lon, double alt, double &x, double &y, double &z){
    //lat = lat*180.0 / PI;
    //lon = lon*180.0 / PI;

//    double r = earthRadius+alt/1000.0;

//    z = r * cos(lat)*cos(lon);
//    x = r * cos(lat)*sin(lon);
//    y = r * sin(lat);


    double cosLat = cos(lat * PI / 180.0);
    double sinLat = sin(lat * PI / 180.0);
    double cosLon = cos(lon * PI / 180.0);
    double sinLon = sin(lon * PI / 180.0);
    double rad = 6378137.0;
    double f = 1.0 / 298.257224;
    double C = 1.0 / sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat);
    double S = (1.0 - f) * (1.0 - f) * C;
    double h =0;
    x = (rad * C + h) * cosLat * cosLon;
    z = (rad * C + h) * cosLat * sinLon;
    y = alt;//(rad * S + h) * sinLat;


}


void getCannyMask(cv::Mat &inputImg, cv::Mat &dispMap, cv::Mat &outputImg){
	// Returns a disparity map with values only on the edges detected by Canny

    cv::Mat contours, grayscale, mask;

    inputImg.convertTo(grayscale, CV_8U);

    cv::blur(grayscale, grayscale, cv::Size(3,3));
    cv::Canny(grayscale, contours, 100, 200);

    contours.convertTo(mask, CV_8U);
    cv::threshold(mask, mask, 100, 255, CV_THRESH_BINARY);
    //cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

    // Masking the Dmap with the Canny mask
    dispMap.copyTo(outputImg, mask);

}

Eigen::Matrix4d translate_rotate(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc, double roll, double pitch, double yaw, Vector3d trans){
    Eigen::Affine3d t;
    t.setIdentity();
    t.translate(trans);

    std::cout <<"TRANSLATION\n" <<t.matrix() << std::endl;

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());

    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d rotationMatrix = q.matrix();
    Eigen::Matrix4d tt = Eigen::Matrix4d::Identity();
    tt.block<3,3>(0,0) = rotationMatrix;

    pcl::transformPointCloud( *pc, *pc, tt);
    pcl::transformPointCloud( *pc, *pc, t.matrix());

    return tt + t.matrix();

}


void getFeaturesMask(cv::Mat &img_0, cv::Mat &img_1, cv::Mat &dispMap_0, cv::Mat &dispMap_1, cv::Mat &featureMap_0, cv::Mat &featureMap_1){

	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 500;
	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute( img_0 , Mat(), keypoints_1, descriptors_1 );
	detector->detectAndCompute( img_1, Mat(), keypoints_2, descriptors_2 );

	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::sort(matches.begin(), matches.end(), compare_DMatches);


	std::vector< DMatch > good_matches = matches;


    good_matches.resize((unsigned long int)std::min((int)good_matches.size(), 72));
    std::cout <<"NB POINT SPARSE : " <<good_matches.size() << std::endl;

	//-- Draw only "good" matches
    Mat img_matches;
	drawMatches( img_0, keypoints_1, img_1, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	imshow("matches", img_matches);
   // cv::waitKey(0);

	//-- Show detected matches
	cv::Mat mask_0(dispMap_0.size(), CV_8U, cv::Scalar(0));
	cv::Mat mask_1(dispMap_1.size(), CV_8U, cv::Scalar(0));


	for( int i = 0; i < (int)good_matches.size(); i++ ){ 
		mask_0.at<uchar>(keypoints_1[ good_matches[i].queryIdx].pt) = 255;
		mask_1.at<uchar>(keypoints_2[ good_matches[i].trainIdx].pt) = 255;

		//printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
	}

    Mat element = getStructuringElement( MORPH_ELLIPSE,
                            Size( 2*3 + 1, 2*3+1 ),
                            Point( 3, 3 ) );
    dilate( mask_0, mask_0, element );

    dilate( mask_1, mask_1, element );


	dispMap_0.copyTo(featureMap_0, mask_0);
	dispMap_1.copyTo(featureMap_1, mask_1);

    Mat tmp;
	vconcat(mask_0, mask_1, tmp);
	imshow("featureMap", tmp);
    waitKey(0);

}

