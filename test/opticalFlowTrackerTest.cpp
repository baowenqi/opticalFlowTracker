#include <iostream>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opticalFlowTracker.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(void)
{
    cout << "hello tracker" << endl;

    Mat rawFrame0 = imread("../data/BlurCar2/img/0001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat rawFrame1 = imread("../data/BlurCar2/img/0002.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    Mat_<float> fltFrame0, fltFrame1;

    rawFrame0.convertTo(fltFrame0, CV_32F);
    rawFrame1.convertTo(fltFrame1, CV_32F);
#if 0
    imshow("frame0", fltFrame0 / 255.0f);
    waitKey(0);

    imshow("frame1", fltFrame1 / 255.0f);
    waitKey(0);
#endif
    ofTracker tracker(fltFrame0.cols, fltFrame0.rows);
    tracker.f_buildPyramid(reinterpret_cast<float*>(fltFrame0.data), tracker.m_imgPyd0);

    ofTracker::box inputBox(227, 207, 122, 99);
    // ofTracker::box outputBox();
    // cout << inputBox << endl;
    // tracker.track(inputBox);

    ofTracker::image tmpImg(fltFrame0.cols, fltFrame0.rows, reinterpret_cast<float*>(fltFrame0.data));
    ofTracker::image tgtImg(fltFrame1.cols, fltFrame1.rows, reinterpret_cast<float*>(fltFrame1.data));

    tracker.f_align(tmpImg, tgtImg, inputBox, inputBox);
#if 0
    for(int i = 3; i >= 0; i--)
    {
        Mat_<float> testOut = Mat(tracker.m_imgPyd0[i]->h, tracker.m_imgPyd0[i]->w, CV_32F, tracker.m_imgPyd0[i]->data);

        imshow("test", testOut / 255.0f);
        waitKey(0);
    }
#endif

    return 0;
}
