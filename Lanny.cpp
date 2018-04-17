#include <iostream>
#include <time.h>
#include "stereo_calib.h"
#include "disparity_map.h"
#include "DepthToPly.h"
#include "parameters.h"
#include "opencv2/surface_matching.hpp"
#include "opencv2/imgproc.hpp"
#include "point_cloud_viewer.h"
#include "tool_box.h"

#include <fstream>
#include <iostream>
#include <dirent.h>

using namespace cv;
using namespace std;

std::vector<Eigen::Matrix4f> load_file(string path){
    std::vector<Eigen::Matrix4f> vec;
    std::ifstream pos_file(path);
    float m00, m01, m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33;
    std::string s;

    while(( pos_file >> m00>> m01>> m02>>m03>> m10>>m11>>m12>>m13>> m20>>m21>>m22>>m23>> m30>>m31>>m32>>m33)){
        Eigen::Matrix4f m;
        m << m00, m01, m02,m23,m10,m11,m12,m13,m20,m21,m22,m03,m30,m31,m32,m33;
        vec.push_back(m);
        std::cout << m << "\n"  <<std::endl;
        std::cout << s << std::endl;
    }
    return vec;
}

float registration(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_source,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_target,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_res,
                   Eigen::Matrix4d& matSourceToTarget ){

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputSource(cloud_source);
    icp.setInputTarget(cloud_target);
    icp.setMaxCorrespondenceDistance (0.5);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations (50);
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon (1e-5);

    icp.align(*cloud_res);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                 icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    matSourceToTarget = icp.getFinalTransformation().cast<double>();
    return icp.getFitnessScore();
}


int main(int argc, char const *argv[]){
    // Read parameters
    std::string images_left_path  = argv[1];
    std::string images_right_path = argv[2];
    std::string positions_path    = argv[3];
    std::string result_path       = argv[4];


    std::vector<Eigen::Matrix4f> vec = load_file(positions_path);

    // Open images directories
    DIR *dir;
    DIR *dir_right;

    if ((dir = opendir (argv[1])) == NULL){
        std::cout << "could not open left images directory" << std::endl;
        return -1;
    }

    if ((dir_right = opendir (argv[2])) == NULL){
        std::cout << "could not open right images directory" << std::endl;
        return -1;
    }

    // Get file names
    struct dirent *ent;
    std::vector<std::string> file_names;
    while ((ent = readdir (dir)) != NULL) {
        file_names.push_back(ent->d_name);
    }
    std::sort(file_names.begin(), file_names.end());
    // Initialisation

    int start_index = 11;

    // Read images
    std::string filenameL  = images_left_path  + file_names.at(start_index);
    std::string filenameR  = images_right_path + file_names.at(start_index);

    cv::Mat left_0  = imread(filenameL);
    cv::Mat right_0 = imread(filenameR);

    if(resize_factor < 1){
        cv::resize(left_0, left_0, cv::Size(), resize_factor, resize_factor);
        cv::resize(right_0, right_0, cv::Size(), resize_factor, resize_factor);
    }

    // First Disparity map
    cv::Mat dispMap_0 = compute_disparity_map(left_0, right_0);

    cv::Mat dispMap_key;
    dispMap_0.copyTo(dispMap_key);


    // Reproject to 3D
    // dense
    cv::Mat points_dense_0;
    cv::reprojectImageTo3D(dispMap_0, points_dense_0, Q, true, -1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_0 = MatToPoinXYZ(points_dense_0, left_0);

    pcl::io::savePLYFileBinary (result_path + "original_dense" +".ply", *cloud_dense_0);


    // Variables initialisation
    cv::Mat left_1, right_1, dispMap_1, points_dense_1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_1    (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_0   (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_1   (new pcl::PointCloud<pcl::PointXYZRGB> ());


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_res    (new pcl::PointCloud<pcl::PointXYZRGB> ());
    cv::Mat points_sparse_1, points_sparse_0;

    int num_iter = 0;
    bool first = true;
    // Loop over all images
    Eigen::Matrix4d matOld;
    matOld.setIdentity();

    for (int i = start_index+1; i < file_names.size(); ++i){

        std::cout << "current file: " + file_names.at(i) << std::endl;

        // Read new files
        filenameL  = images_left_path  + file_names.at(i);
        filenameR  = images_right_path + file_names.at(i);
        left_1  = imread(filenameL);
        right_1 = imread(filenameR);

        if(! left_1.data)
            continue;

        dispMap_1 = compute_disparity_map(left_1, right_1);

        //-- Reprojection sparse
        cv::Mat featureMap_0, featureMap_1;
        getFeaturesMask(left_0, left_1, dispMap_0, dispMap_1, featureMap_0, featureMap_1);

        cv::reprojectImageTo3D(featureMap_0, points_sparse_0, Q, true, -1);
        cv::reprojectImageTo3D(featureMap_1, points_sparse_1, Q, true, -1);

        cloud_sparse_0 = MatToPoinXYZ(points_sparse_0, left_0);
        cloud_sparse_1 = MatToPoinXYZ(points_sparse_1, left_1);


        pcl::transformPointCloud( *cloud_sparse_0, *cloud_sparse_0, vec[i-3]);
        pcl::transformPointCloud( *cloud_sparse_0, *cloud_sparse_0, matOld);

        first = false;


        cv::reprojectImageTo3D(dispMap_1, points_dense_1, Q, true, -1);
        cloud_dense_1 = MatToPoinXYZ(points_dense_1, left_1);

        pcl::transformPointCloud( *cloud_dense_1, *cloud_dense_1, vec[i-2]);
        pcl::transformPointCloud( *cloud_sparse_1, *cloud_sparse_1, vec[i-2]);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_res  (new pcl::PointCloud<pcl::PointXYZRGB>);
        float res = registration(cloud_sparse_1, cloud_sparse_0, cloud_res, matOld);


        if(display3D){
            // Initialise viewer
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(0,0,0);
            viewer->addCoordinateSystem(0.1);

            // Color point clouds
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color1 (cloud_sparse_1,  20, 100, 200);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color2 (cloud_sparse_0  , 200, 50 , 50 );

            viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sparse_1, single_color1, "sample_cloud_1");
            viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sparse_0  , single_color2, "sample_cloud_2");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample_cloud_1");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample_cloud_2");

            // Viewer loop
            while(!viewer->wasStopped()){
                viewer->spinOnce();
                boost::this_thread::sleep (boost::posix_time::microseconds(100000));
            }

        }


        dispMap_0 = dispMap_1;
        left_0 = left_1;

        pcl::transformPointCloud( *cloud_dense_1 , *cloud_dense_1 , matOld);

        std::cout << vec[i-2] << std::endl;
        *cloud_dense_res += *cloud_dense_1;

        pcl::io::savePLYFileBinary (result_path + "Final_dense_" + std::to_string(num_iter++) +".ply", *cloud_dense_res);
    }
    

    return 0;
}
