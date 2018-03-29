
#include "point_cloud_viewer.h"


void matToPointXYZ(cv::Mat &color, cv::Mat &depth,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                   const double u0, const double v0, const double f ) {

  const double fx = f;
  const double fy = f;
  int rows = color.rows;
  int cols = color.cols;
  cloud->height = (uint32_t) rows;
  cloud->width = (uint32_t) cols;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width * cloud->height);
  for (unsigned int u = 0; u < rows; ++u) {
    for (unsigned int v = 0; v < cols; ++v) {
      float Xw = 0, Yw = 0, Zw = 0;

      Zw = depth.at<ushort>(u, v);
      Xw = (float) ((v - v0) * Zw / fx);
      Yw = (float) ((u - u0) * Zw / fy);

      cloud->at(v, u).b = color.at<cv::Vec3b>(u, v)[0];
      cloud->at(v, u).g = color.at<cv::Vec3b>(u, v)[1];
      cloud->at(v, u).r = color.at<cv::Vec3b>(u, v)[2];
      cloud->at(v, u).x = Xw;
      cloud->at(v, u).y = Yw;
      cloud->at(v, u).z = Zw;
    }
  }
}


/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr VectorToPCL(cv::vector<Point2f> points){

     pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

}*/


pcl::PointCloud<pcl::PointXYZRGB>::Ptr MatToPoinXYZ(cv::Mat OpencVPointCloud, cv::Mat color){


     pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

     for(int i=0;i<OpencVPointCloud.rows;i++){
         for(int j=0;j<OpencVPointCloud.cols;j++){

                  if(abs((float)OpencVPointCloud.at<cv::Vec3f>(i, j)[2]) < 5 &&
                     abs((float)OpencVPointCloud.at<cv::Vec3f>(i, j)[2]) > 0){
                   pcl::PointXYZRGB point;
                   point.x =  (float)OpencVPointCloud.at<cv::Vec3f>(i, j)[0];
                   point.y =  (float)OpencVPointCloud.at<cv::Vec3f>(i, j)[1];
                   point.z = -1.0*(float)OpencVPointCloud.at<cv::Vec3f>(i, j)[2];
                   point.r = (float)color.at<cv::Vec3b>(i, j)[2];
                   point.g = (float)color.at<cv::Vec3b>(i, j)[1];
                   point.b = (float)color.at<cv::Vec3b>(i, j)[0];


                   point_cloud_ptr -> points.push_back(point);
                 }
                 //std::cout << point.x << " " << point.y << " " << point.z << std::endl;


            //std::cout << point.x << " " << point.y << " " << point.z << std::endl;

            // when color needs to be added:
            //uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
            //point.rgb = *reinterpret_cast<float*>(&rgb);


         }
     }
     point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
     point_cloud_ptr->height = 1;

     return point_cloud_ptr;
 }


// Reproject image to 3D
 pcl::PointCloud<pcl::PointXYZ>::Ptr customReproject(const cv::Mat& recons3D, const cv::Mat& img){

     pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);//(new pcl::pointcloud<pcl::pointXYZ>);

     for (int y = 0; y < recons3D.rows; y++){
                 for (int x = 0; x < recons3D.cols; x++){
                     cv::Point3f pointOcv = recons3D.at<cv::Point3f>(y, x);

                         //Insert info into point cloud structure
                         pcl::PointXYZ point;
                         point.x = -pointOcv.x;  // xyz points transformed in Z upwards system
                         point.y = -pointOcv.z;
                         point.z =  pointOcv.y;
                         //r = g = b = imgLgray.at<uchar>(y,x);  // rgb vals form left image gray cause i hav no color cam
                         //uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
                         //point.rgb = *reinterpret_cast<float*>(&rgb);
                         point_cloud_ptr->points.push_back (point);  // pushback actual point

                  }
     }
     point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
     point_cloud_ptr->height = 1;
     return point_cloud_ptr;
}
