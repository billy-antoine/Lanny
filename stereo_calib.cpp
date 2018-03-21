
#include "stereo_calib.h"

int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << 0 << ") opening " << dir << std::endl;
        return 0;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(dir + std::string(dirp->d_name));
    }
    closedir(dp);
    std::sort( files.begin(), files.end() );
    files.erase(files.begin());
    files.erase(files.begin());
    return 0;
}

void loadIntrinsics(const std::string path, cv::Mat &M1, cv::Mat &M2, cv::Mat &D1, cv::Mat &D2)
{ // Read intrinsics parameters from file
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["M1"] >> M1;
    fs["M2"] >> M2;
    fs["D1"] >> D1;
    fs["D2"] >> D2;

    fs.release();
}

void loadExtrinsics(const std::string path, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &R, cv::Mat &T, cv::Mat &Q)
{ // Read extrinsics parameters from file
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["Q"] >> Q;

    fs.release();
}

void loadRoIs(const std::string path, cv::Rect &roi1, cv::Rect &roi2)
{ // Read ROIs parameters from file
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["Roi1"] >> roi1;
    fs["Roi2"] >> roi2;

    fs.release();
}
std::vector<std::string> getCalibrationData(int num_cam1, int num_cam2, const char* imagelistfn)
{
    cv::VideoCapture capture1(num_cam1);  //Capture using any camera connected to your system
    if(!capture1.isOpened())
        std::cout << "can't read input camera" << std::endl;

    cv::VideoCapture capture2(num_cam2);  //Capture using any camera connected to your system
    if(!capture2.isOpened())
        std::cout << "can't read input camera" << std::endl;

    capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 960);

    capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 960);

    cv::Mat img_1, img_2;

    int i = 0;
    cv::namedWindow("Stereo Vision");

    // Waiting window
    while (true){
        capture1.read(img_1);
        capture2.read(img_2);

        cv::transpose(img_1,img_1);
        cv::transpose(img_2,img_2);

        /*capture1 >> img_1;
        capture2 >> img_2;
        img_1 = img_1.clone();
        img_2 = img_2.clone();*/

        // Display images
        cv::Mat canvas, img_12, img_22;
        img_12 = img_1.clone();
        img_22 = img_2.clone();

       /* std::vector<cv::Point2f> corners;
        bool found = findChessboardCorners(img_1, boardSize, corners,
                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(found)
            drawChessboardCorners(img_12, boardSize, corners, found);


        bool found2 = findChessboardCorners(img_2, boardSize, corners,
                                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(found2)
            drawChessboardCorners(img_22, boardSize, corners, found);
        */
        cv::hconcat(img_12, img_22, canvas);
        cv::putText(canvas, "Press SPACE to take snapshot", cv::Point(25, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(30, 200, 30), 1, CV_AA);

        imshow("Stereo Vision", canvas);

        char k = cvWaitKey(10);
        if (char(k) == 32 /*&& found && found2*/){ // if 'SPACE' is pressed we save the images pair
            cv::threshold(canvas, canvas, 70, 255, CV_THRESH_BINARY_INV);
            imshow("Stereo Vision", canvas);
            cv::waitKey(50);
            std::string name = std::string(imagelistfn) + (i < 10 ? "0" : "") + std::to_string(i) + "left.png";
            std::string name2 = std::string(imagelistfn) + (i < 10 ? "0" : "") + std::to_string(i) + "right.png";
            ++i;
            imwrite(name, img_1);
            imwrite(name2, img_2);
        }
        else if (char(k) == 100) // if 'd' is pressed we end the writing
            break;
    }

    // Get image names
    std::vector<std::string> imagelist;
    getdir(imagelistfn, imagelist);

    return imagelist;
}


void StereoCalib(const std::vector<std::string>& imagelist, std::__cxx11::string outputFolder, cv::Size boardSize, float squareSize, bool displayCorners, bool useCalibrated, bool showRectified)
{
    if (imagelist.size() % 2 != 0)
    {
        std::cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    // ARRAY AND VECTOR STORAGE:

    std::vector<std::vector<cv::Point2f> > imagePoints[2];
    std::vector<std::vector<cv::Point3f> > objectPoints;
    cv::Size imageSize;

    int i, j, k, nimages = (int)imagelist.size() / 2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    std::vector<std::string> goodImageList;

    for (i = j = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            const std::string& filename = imagelist[i * 2 + k];
            cv::Mat img = cv::imread(filename, 0);
            if (img.empty())
                break;
            if (imageSize == cv::Size())
                imageSize = img.size();
            else if (img.size() != imageSize)
            {
                std::cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            std::vector<cv::Point2f>& corners = imagePoints[k][j];
            for (int scale = 1; scale <= maxScale; scale++)
            {
                cv::Mat timg;
                if (scale == 1)
                    timg = img;
                else
                    cv::resize(img, timg, cv::Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
                if (found)
                {
                    if (scale > 1)
                    {
                        cv::Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
            }
            if (displayCorners)
            {
                std::cout << filename << std::endl;
                cv::Mat cimg, cimg1;
                cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640. / MAX(img.rows, img.cols);
                resize(cimg, cimg1, cv::Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)cv::waitKey(500);
                if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if (!found)
                break;
            cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                30, 0.01));
        }
        if (k == 2)
        {
            goodImageList.push_back(imagelist[i * 2]);
            goodImageList.push_back(imagelist[i * 2 + 1]);
            j++;
        }
    }
    std::cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if (nimages < 2)
    {
        std::cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for (i = 0; i < nimages; i++)
    {
        for (j = 0; j < boardSize.height; j++)
            for (k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(cv::Point3f(k*squareSize, j*squareSize, 0));
    }

    std::cout << "Running stereo calibration ...\n";

    cv::Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    cv::Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, E, F,
        cv::CALIB_FIX_ASPECT_RATIO +
        cv::CALIB_ZERO_TANGENT_DIST +
        cv::CALIB_USE_INTRINSIC_GUESS +
        cv::CALIB_SAME_FOCAL_LENGTH +
        cv::CALIB_RATIONAL_MODEL +
        cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
    std::cout << "done with RMS error=" << rms << std::endl;

    // CALIBRATION QUALITY CHECK
    // because the output fundamental matrix implicitly
    // includes all the output information,
    // we can check the quality of calibration using the
    // epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    std::vector<cv::Vec3f> lines[2];
    for (i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        cv::Mat imgpt[2];
        for (k = 0; k < 2; k++)
        {
            imgpt[k] = cv::Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    std::cout << "average epipolar err = " << err / npoints << std::endl;

    // save intrinsic parameters
    cv::FileStorage fs(outputFolder+"intrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        std::cout << "Error: can not save the intrinsic parameters\n";

    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open(outputFolder+"extrinsics.yml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        std::cout << "Error: can not save the extrinsic parameters\n";


    fs.open(outputFolder+"Rois.yml", cv::FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "Roi1" << validRoi[0] << "Roi2" << validRoi[1];
        fs.release();
    }
    else
        std::cout << "Error: can not save the Rois\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    // COMPUTE AND DISPLAY RECTIFICATION
    if (!showRectified)
        return;

    cv::Mat rmap[2][2];
    // IF BY CALIBRATED (BOUGUET'S METHOD)
    if (useCalibrated)
    {
        // we already computed everything
    }
    // OR ELSE HARTLEY'S METHOD
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        std::vector<cv::Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = cv::findFundamentalMat(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), cv::FM_8POINT, 0, 0);
        cv::Mat H1, H2;
        stereoRectifyUncalibrated(cv::Mat(allimgpt[0]), cv::Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];

        fs.open(outputFolder+"extrinsics.yml", cv::FileStorage::WRITE);
        if (fs.isOpened())
        {
            fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
            fs.release();
        }
        else
            std::cout << "Error: can not save the extrinsic parameters\n";
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    cv::Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h * 2, w, CV_8UC3);
    }

    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            cv::Mat img = cv::imread(goodImageList[i * 2 + k], 0), rimg, cimg;
            cv::remap(img, rimg, rmap[k][0], rmap[k][1], cv::INTER_LINEAR);
            cvtColor(rimg, cimg, cv::COLOR_GRAY2BGR);
            cv::Mat canvasPart = !isVerticalStereo ? canvas(cv::Rect(w*k, 0, w, h)) : canvas(cv::Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, cv::INTER_AREA);
            if (useCalibrated)
            {
                cv::Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                    cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, cv::Scalar(0, 0, 255), 3, 8);
            }
        }

        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)cv::waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}
