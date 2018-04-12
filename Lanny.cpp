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

int main(int argc, char const *argv[]){

    // Read parameters
    std::string images_left_path  = argv[1];
    std::string images_right_path = argv[2];
    std::string positions_path    = argv[3];
    std::string result_path       = argv[4];

    // Open images directories
    DIR *dir;
    DIR *dir_right;
    DIR *dir_pos;

    if ((dir = opendir (argv[1])) == NULL){
        std::cout << "could not open left images directory" << std::endl;
        return -1;
    } 

    if ((dir_right = opendir (argv[2])) == NULL){
        std::cout << "could not open right images directory" << std::endl;
        return -1;
    } 

    if ((dir_pos = opendir (argv[3])) == NULL){
        std::cout << "could not open positions directory" << std::endl;
        return -1;
    }
    closedir(dir_right);

    // Get file names
    struct dirent *ent;
    std::vector<std::string> file_names;
    while ((ent = readdir (dir)) != NULL) {
        file_names.push_back(ent->d_name);
    }

    // Get positions file names
    struct dirent *ent2;
    std::vector<std::string> positions_names;
    while ((ent2 = readdir (dir_pos)) != NULL) {
        positions_names.push_back(ent2->d_name);
    }
    closedir(dir_pos);

    std::sort(file_names.begin(), file_names.end());
    std::sort(positions_names.begin(), positions_names.end());
    // Initialisation

    // Read images
    std::string filenameL  = images_left_path  + file_names.at(2);
    std::string filenameR  = images_right_path + file_names.at(2);

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

    // Display
    if(displayImages){
        cv::Mat dispN;
        cv::normalize(dispMap_0, dispN,  0, 255, CV_MINMAX, CV_8U);
        cv::applyColorMap( dispN, dispN, COLORMAP_PARULA );
        cv::imshow("disp map", dispN);
        cv::waitKey(50);
    }


    // Reproject to 3D
    // dense
    cv::Mat points_dense_0, points_sparse_0;
    cv::reprojectImageTo3D(dispMap_0, points_dense_0, Q, true, -1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_0 = MatToPoinXYZ(points_dense_0, left_0);


    // Read positions
    std::ifstream pos_file(positions_path+positions_names.at(2));
    double lat, lon, alt, roll, pitch, yaw;
    if(!(pos_file >> lat >> lon >> alt >> roll >> pitch >> yaw))
        cout << "could not read positions" << endl;
    pos_file.close();


    // Apply GPS transformation
    double x,y,z, xf, yf, zf;
    llaToXyz(lat, lon, alt, xf, yf, zf);

    translate_rotate(cloud_dense_0, roll, pitch, yaw, Eigen::Vector3d(0,0,0));

    pcl::io::savePLYFileBinary (result_path + "original_dense" +".ply", *cloud_dense_0);

    // sparse
    //cv::Mat cannied_0;
    //getCannyMask(left_0, dispMap_0, cannied_0);
    //cv::reprojectImageTo3D(cannied_0, points_sparse_0, Q, true, -1);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_0 = MatToPoinXYZ(points_sparse_0, left_0);


    // Variables initialisation
    cv::Mat left_1, right_1, dispMap_1, points_dense_1, cannied_1, points_sparse_1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_0   (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_1    (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_1_t  (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_1   (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sparse_1_t (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr key_sparse       (new pcl::PointCloud<pcl::PointXYZRGB> ());

    // Final point clouds
    pcl::PointCloud<pcl::PointXYZRGB> Final_sparse = *cloud_sparse_0;
    pcl::PointCloud<pcl::PointXYZRGB> Final_dense  = *cloud_dense_0;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_sparse_ptr(&Final_sparse);


    // Transformation matrice
    Eigen::Matrix4f transformationMatrix_old = Eigen::Matrix4f::Identity ();

    int num_iter = 0;

    // Loop over all images
    for (int i = 3; i < file_names.size(); ++i){

        std::cout << "current file: " + file_names.at(i) << std::endl;

        // Read new files
        filenameL  = images_left_path  + file_names.at(i);
        filenameR  = images_right_path + file_names.at(i);
        left_1  = imread(filenameL);
        right_1 = imread(filenameR);

        if(resize_factor < 1){
        cv::resize(left_1, left_1, cv::Size(), resize_factor, resize_factor);
        cv::resize(right_1, right_1, cv::Size(), resize_factor, resize_factor);
        }

        if(! left_1.data)
            continue;

        // Disparity
        dispMap_1 = compute_disparity_map(left_1, right_1);

        if(displayImages){
            cv::Mat disp_vis;
            cv::normalize(dispMap_1, disp_vis,  0, 255, CV_MINMAX, CV_8U);
            cv::applyColorMap( disp_vis, disp_vis, COLORMAP_PARULA );
            cv::imshow("disp map", disp_vis);
            cv::waitKey(50);
        }

        /*
        //-- Reprojection sparse
        cv::Mat featureMap_0, featureMap_1;
        getFeaturesMask(left_0, left_1, dispMap_0, dispMap_1, featureMap_0, featureMap_1);

        cv::reprojectImageTo3D(featureMap_0, points_sparse_0, Q, true, -1);
        cv::reprojectImageTo3D(featureMap_1, points_sparse_1, Q, true, -1);

        cloud_sparse_0 = MatToPoinXYZ(points_sparse_0, left_0);
        cloud_sparse_1 = MatToPoinXYZ(points_sparse_1, left_1);

        //-- DisparityMaps fusion
        //-- ICP
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setTransformationEpsilon (1e-8);
        icp.setMaxCorrespondenceDistance (0.5);
        icp.setMaximumIterations (50);

        icp.setInputSource(cloud_sparse_1);
        icp.setInputTarget(cloud_sparse_0);

        // Magic happens here
        icp.align(*final_sparse_ptr);        
        std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    
        // Get new transformation matrice (concatenation by *)
        Eigen::Matrix4f transformationMatrix_new =  icp.getFinalTransformation ();

        std::cout<<"trans : \n"<<transformationMatrix_new<<std::endl;

        // Transform sparse PC
        pcl::transformPointCloud( *cloud_sparse_1, *cloud_sparse_1_t, transformationMatrix_new); 


        if(num_iter%densification_ratio == 0){        
            //pcl::PointCloud<pcl::PointXYZRGB> merged_dense = *cloud_dense_0;
            //merged_dense += *cloud_dense_1_t;

            cv::reprojectImageTo3D(dispMap_1, points_dense_1, Q, true, -1);
            cloud_dense_1 = MatToPoinXYZ(points_dense_1, left_1);  


            // Get new transformation matrice (concatenation by *)
            Eigen::Matrix4f transformationMatrix_global =  transformationMatrix_old * transformationMatrix_new;

            // Transform Dense PC
            pcl::transformPointCloud( *cloud_dense_1 , *cloud_dense_1_t , transformationMatrix_global);

            // Apply the old transformation for the final global point cloud
            //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dense_1_t2 (new pcl::PointCloud<pcl::PointXYZRGB> ());
            //pcl::transformPointCloud( *cloud_dense_1_t , *cloud_dense_1_t2 , transformationMatrix_old);
            Final_dense += *cloud_dense_1_t;

            // PC savings
            //pcl::io::savePLYFileBinary (result_path + "merged_sparse_" + std::to_string(num_iter) +".ply", *final_sparse_ptr);
            //pcl::io::savePLYFileBinary (result_path + "merged_dense_" + std::to_string(num_iter) +".ply", merged_dense);
            pcl::io::savePLYFileBinary (result_path + "Final_dense_" + std::to_string(num_iter) +".ply", Final_dense);

        }*/


        cv::reprojectImageTo3D(dispMap_1, points_dense_1, Q, true, -1);
        cloud_dense_1 = MatToPoinXYZ(points_dense_1, left_1);

        // Read positions

        std::ifstream pos_file_1(positions_path+positions_names.at(i));

        if(!(pos_file_1 >> lat >> lon >> alt >> roll >> pitch >> yaw))
            cout << "could not read positions" << endl;
        pos_file_1.close();

        // Apply GPS transformation
        llaToXyz(lat, lon, alt, x, y, z);
        translate_rotate(cloud_dense_1, roll, pitch, yaw, Eigen::Vector3d ((x-xf) ,(y-yf),(z-zf)));


        // Vizualisation
        if(display3D){
            // Initialise viewer
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(0,0,0);
            viewer->addCoordinateSystem(0.1);            

            // Color point clouds
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color1 (cloud_sparse_1_t,  20, 100, 200);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color2 (cloud_sparse_0  , 200, 50 , 50 );

            viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sparse_1_t, single_color1, "sample_cloud_1");
            viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sparse_0  , single_color2, "sample_cloud_2");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample_cloud_1");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample_cloud_2");

            // Viewer loop
            while(!viewer->wasStopped()){
                  viewer->spinOnce();
                  boost::this_thread::sleep (boost::posix_time::microseconds(100000));
            }

        }

        // Switch variables
        //transformationMatrix_old *= transformationMatrix_new;

        /*if(num_iter%densification_ratio == 0){
            pcl::copyPointCloud(*cloud_dense_1, *cloud_dense_0);
        }

        pcl::copyPointCloud(*cloud_sparse_1, *cloud_sparse_0);
        left_1.copyTo(left_0);
        dispMap_1.copyTo(dispMap_0);
        num_iter++;*/
        *cloud_dense_0 += *cloud_dense_1;
        pcl::io::savePLYFileBinary (result_path + "Final_dense_" + std::to_string(num_iter++) +".ply", *cloud_dense_0);


    }
    closedir (dir);
    

    return 0;
}
