#include <iostream>
#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opticalFlowTracker.hpp"

using namespace std;
using namespace cv;

int main(void)
{
    cout << "hello tracker" << endl;

    Mat rawFrame0 = imread("../data/BlurCar2/img/0001.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    Mat_<float> fltFrame0;

    rawFrame0.convertTo(fltFrame0, CV_32F);

    imshow("frame0", fltFrame0 / 255.0f);
    waitKey(0);

    ofTracker tracker(fltFrame0.cols, fltFrame0.rows);
    tracker.f_buildPyramid(reinterpret_cast<float*>(fltFrame0.data), tracker.m_imgPyd0);

    ofTracker::box inputBox(227, 207, 122, 99);
    cout << inputBox << endl;
    tracker.track(inputBox);

    ofTracker::matrix<float> a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;

    ofTracker::matrix<float> b(3, 2);
    b << 1, 2,
         3, 4,
         5, 6;

    ofTracker::matrix<float> c(2, 2);
    c = a * b;

    cout << a << endl;
    cout << b << endl;
    cout << c << endl;

    for(int i = 3; i >= 0; i--)
    {
        Mat_<float> testOut = Mat(tracker.m_imgPyd0[i]->h, tracker.m_imgPyd0[i]->w, CV_32F, tracker.m_imgPyd0[i]->data);

        imshow("test", testOut / 255.0f);
        waitKey(0);
    }

    return 0;
}
