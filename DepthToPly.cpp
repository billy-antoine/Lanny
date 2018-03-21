/*
 * DepthToPly.cpp
 *
 *  Created on: 24 août 2017
 *      Author: abilly
 */


#include "DepthToPly.h"

using namespace cv;
using namespace std;


void savePLY(String fileName, Mat &points, Mat &intensities, int MaxValue)
{
    // Save in .ply with projected intensity

    // Estimate number of points
    int number_of_points = 0;
    for (int i = 0; i < points.rows; i++) {
        for (int j = 0; j < points.cols; j++) {
            if (abs(points.at<Vec3f>(i, j)[2]) < MaxValue) {
                number_of_points++;
            }
        }
    }

    stringstream str(stringstream::out|stringstream::binary);

    str << "ply" << endl;
    str << "format ascii 1.0" << endl;
    str << "element vertex " << number_of_points << endl;
    str << "property float x" << endl;
    str << "property float y" << endl;
    str << "property float z" << endl;
    //file.Write("property double confidence\n");
    str << "property uchar red" << endl;
    str << "property uchar green" << endl;
    str << "property uchar blue" << endl;
    str << "end_header" << endl;


    for (int i = 0; i < points.rows; i++) {
        for (int j = 0; j < points.cols; j++) {
            if (abs(points.at<Vec3f>(i, j)[2]) < MaxValue) {
                str << comma_sep << (float)points.at<Vec3f>(i, j)[0]   << " " << comma_sep << (float)points.at<Vec3f>(i, j)[1] << " " << comma_sep << -1.0*(float)points.at<Vec3f>(i, j)[2];
                str << " " << (float)intensities.at<Vec3b>(i, j)[2] << " " << (float)intensities.at<Vec3b>(i, j)[1] << " " << (float)intensities.at<Vec3b>(i, j)[0] << endl;
            }
        }
    }

    ofstream point_cloud_file;
    point_cloud_file.open(fileName.c_str());
    point_cloud_file.write(str.str().c_str(), str.str().length());
    point_cloud_file.close();
}

void saveXYZ(const char* filename, const Mat& mat)
{
    // quick writting in .xyz format
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for (int y = 0; y < mat.rows; y++){
        for (int x = 0; x < mat.cols; x++)   {
            Vec3f point = mat.at<Vec3f>(y, x);
                if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
                    fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

void writeMatToFile(cv::Mat& m, String filename)
{
    ofstream fout(filename);

    if(!fout){
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++){
        for(int j=0; j<m.cols; j++){
            //if(abs(m.at<float>(i,j)) < 100)
                fout << i << " " << j << " " << m.at<float>(i,j)<<"\n";
        }
        //fout<<endl;
    }

    fout.close();
}
