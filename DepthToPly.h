#ifndef DEPTH_TO_PLY
#define DEPTH_TO_PLY

#include <opencv2/opencv.hpp>
#include "dirent.h"
#include <locale>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <cctype>

#include <fstream>

template <typename T>
struct comma_separator : std::numpunct<T>
{
    typename std::numpunct<T>::char_type do_decimal_point() const
    {
        return ',';
    }
};

template <typename T>
std::basic_ostream<T>& comma_sep(std::basic_ostream<T>& os)
{
    os.imbue(std::locale(std::locale(""), new comma_separator<T>));
    return os;
}

void savePLY(cv::String fileName, cv::Mat &points, cv::Mat &intensities, int MaxValue = 500);
void saveXYZ(const char* filename, const cv::Mat& mat);

void writeMatToFile(cv::Mat& m, cv::String filename);

#endif
