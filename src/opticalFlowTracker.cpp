#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <list>
#include <algorithm>
#include <iostream>
#include <sys/time.h>
#include "opticalFlowTracker.hpp"

using namespace std;

ofTracker::ofTracker
(
    int32_t i_imgWidth,
    int32_t i_imgHeight
) :
m_imgWidth(i_imgWidth),
m_imgHeight(i_imgHeight)
{
#if 0
    // ---------------------------------- //
    // allocate the warping matrix        //
    // in this example, we set the final  //
    // warping model as similarity with   //
    // 3 dof:                             //
    // 1+s, 0,   t1,                      //
    // 0,   1+s, t2,                      //
    // ---------------------------------- //
    outW = new float[m_dof * 2];

    // ---------------------------------- //
    // setup template and target pyramids //
    // each has 4 level, scale down by 2  //
    // ---------------------------------- //
    for(int i = (m_numPyramid - 1); i >= 0; i--)
    {
        int32_t imgWidth, imgHeight;

        if(i == m_numPyramid - 1)
        {
            imgWidth = m_imgWidth;
            imgHeight = m_imgHeight;
        }
        else
        {
            imgWidth = m_imgPyd0[i + 1]->w >> 1;
            imgHeight = m_imgPyd0[i + 1]->h >> 1;
        }

        m_imgPyd0[i] = new image(imgWidth, imgHeight);
        m_imgPyd1[i] = new image(imgWidth, imgHeight);

        m_boxPyd[i] = new box;
    }
#endif
}

ofTracker::~ofTracker()
{
#if 0
    if(outW != nullptr) delete []outW;

    for(int i = (m_numPyramid - 1); i >= 0; i--)
    {
        if(m_imgPyd0[i] != nullptr) delete m_imgPyd0[i];
        if(m_imgPyd1[i] != nullptr) delete m_imgPyd1[i];
        if(m_boxPyd[i] != nullptr) delete m_boxPyd[i];
    }
#endif
}

ofTracker::status_t ofTracker::f_convolution
(
    const matrix<float>&    src,
    const matrix<float>&    knl,
          matrix<float>&    dst
)
{
    assert(src.cols == dst.cols && src.rows == dst.rows);

    int32_t kwRad = (knl.cols >> 1);
    int32_t khRad = (knl.rows >> 1);
    int32_t wPad = src.cols + kwRad * 2;
    int32_t hPad = src.rows + khRad * 2;

    // ---------------------------------------------- //
    // pad the boundry of the src image with 0s       //
    // ---------------------------------------------- //
    float* srcPad = new float[wPad * hPad];
    memset(srcPad, 0, wPad * hPad * sizeof(float));
    for(int y = 0; y < src.rows; y++)
    {
        for(int x = 0; x < src.cols; x++)
        {
            int srcAddr = y * src.cols + x;
            int padAddr = (y + khRad) * wPad + (x + kwRad);
            *(srcPad + padAddr) = *(src.data + srcAddr);
        }
    }

    // ---------------------------------------------- //
    // convolve the srcPad with knl                   //
    // ---------------------------------------------- //
    for(int y = 0; y < dst.rows; y++)
    {
        for(int x = 0; x < dst.cols; x++)
        {
            float result = 0.0f;
            for(int v = 0; v < knl.rows; v++)
            {
                for(int u = 0; u < knl.cols; u++)
                {
                    int knlAddr = v * knl.cols + u;
                    int srcAddr = (y + v) * wPad + (x + u);
                    
                    float knlData = *(knl.data + knlAddr);
                    float srcData = *(srcPad + srcAddr);
                    result += srcData * knlData;
                }
            }
            int dstAddr = y * dst.cols + x;
            *(dst.data + dstAddr) = result;
        }
    }

    delete []srcPad;
    return SUCCESS;
}

#if 0
ofTracker::status_t ofTracker::f_buildPyramid
(
    float* frame, 
    image* (&pyramid)[m_numPyramid]
)
{
    // ---------------------------------------------- //
    // gaussian filter kernel                         //
    // ---------------------------------------------- //
    float knl[9] = {0.0625, 0.125, 0.0625, \
                    0.125,  0.25,  0.125,  \
                    0.0625, 0.125, 0.0625  };
    image knlImg(3, 3, knl);

    // ---------------------------------------------- //
    // the bottom level is just a copy of input frame //
    // the upper-3 level is down-scaled by 2 each     //
    // ---------------------------------------------- //
    for(int l = m_numPyramid - 1; l >= 0; l--)
    {
        if(l == m_numPyramid - 1)
        {
            memcpy(pyramid[l]->data, frame, pyramid[l]->w * pyramid[l]->h * sizeof(float));
        }
        else
        {
            image gaussianImg(pyramid[l+1]->w, pyramid[l+1]->h);

            f_convolution(*(pyramid[l+1]), knlImg, gaussianImg);

            for(int y = 0; y < pyramid[l]->h; y++)
            {
                float fltY = y * 2.0f + 0.5f;
                for(int x = 0; x < pyramid[l]->w; x++)
                {
                    float fltX = x * 2.0f + 0.5f;
                    float sampleData = f_sample(gaussianImg, fltX, fltY);

                    int addr = y * pyramid[l]->w + x;
                    *(pyramid[l]->data + addr) = sampleData; 
                }
            }
        }
    }

    return SUCCESS;
}
#endif

ofTracker::status_t ofTracker::track
(
    box&    inputBox
)
{
    *(m_boxPyd[0]) = inputBox * 0.125f;
    cout << *(m_boxPyd[0]) << endl;

    return SUCCESS;
}

ofTracker::status_t ofTracker::f_align
(
    matrix<float>&   tmpImg,
    matrix<float>&   tgtImg,
    box&             tmpBox,
    box&             tgtBox
)
{
    // ----------------------------------------------- //
    // initial the warping matrix                      //
    // we use scaling + translation model:             //
    // 1+s, 0,   b1,                                   //
    // 0,   1+s, b2                                    //
    // ----------------------------------------------- //
    matrix<float> W(2, 3, {1, 0, 0,
                           0, 1, 0});

    // ----------------------------------------------- //
    // initial the hessian matrix                      //
    // ----------------------------------------------- //
    matrix<float> H(3, 3);
    matrix<float> invH(3, 3);

    // ----------------------------------------------- //
    // STEP 3                                          //
    // compute the gradient of template image          //
    // we use sobel filter here                        //
    // ----------------------------------------------- //
    matrix<float> sobelX(3, 3, {-1,  0,  1,
                                -2,  0,  2,
                                -1,  0,  1});

    matrix<float> sobelY(3, 3, {-1, -2, -1,
                                 0,  0,  0,
                                 1,  2,  1});

    matrix<float> tmpImgGx(tmpImg.rows, tmpImg.cols);
    matrix<float> tmpImgGy(tmpImg.rows, tmpImg.cols);

    f_convolution(tmpImg, sobelX, tmpImgGx);
    f_convolution(tmpImg, sobelY, tmpImgGy);

    // ----------------------------------------------- //
    // STEP 4                                          //
    // pre compute the Jacobian for given rect         //
    // STEP 5                                          //
    // compute the steepest descent images             //
    // STEP 6                                          //
    // compute the hessian matrix                      //
    // accumulate to H                                 //
    // compute the inverse of H                        //
    // ----------------------------------------------- //
    matrix<float>* sdiTArray[tmpBox.h][tmpBox.w];

    for(int v = 0; v < tmpBox.h; v++)
    {
        for(int u = 0; u < tmpBox.w; u++)
        {
            matrix<float> jacobian(2, 3);
            jacobian << u, 1, 0,
                        v, 0, 1;

            float gx = tmpImgGx.at(tmpBox.y + v, tmpBox.x + u);
            float gy = tmpImgGy.at(tmpBox.y + v, tmpBox.x + u);

            matrix<float> gradient(1, 2, {gx, gy});

            matrix<float> sdi = gradient * jacobian;

            matrix<float> sdiT = sdi.transpose();

            matrix<float> h = sdiT * sdi;
            
            H += h;
            
            sdiTArray[v][u] = new matrix<float>(3, 1);
            *(sdiTArray[v][u]) = sdiT;
        }
    }
    invH = H.inverse();

    // --------------------------------------------------- //
    // iterating...                                        //
    // --------------------------------------------------- //
    for(int i = 0; i < 500; i++)
    {
        // ----------------------------------------------- //
        // initial the error-weigthed sdi                  //
        // ----------------------------------------------- //
        matrix<float> S(3, 1);

        for(int v = 0; v < tmpBox.h; v++)
        {
            for(int u = 0; u < tmpBox.w; u++)
            {
                // ----------------------------------------------- //
                // STEP 1                                          //
                // compute the warp image                          //
                // STEP 2                                          //
                // compute the error image                         //
                // ----------------------------------------------- //
                matrix<float> x(3, 1);
                x << u, v, 1;

                matrix<float> wx = W * x;
                float uw = wx.at(0);
                float vw = wx.at(1);

                float tgtData = tgtImg.at(tmpBox.y + vw, tmpBox.x + uw);
                float tmpData = tmpImg.at(tmpBox.y + v,  tmpBox.x + u );
                float error = tgtData - tmpData;

                // --------------------------------------- //
                // STEP 7                                  //
                // aggregate the error-weigthed sdi        //
                // --------------------------------------- //
                matrix<float> sdiT = *(sdiTArray[v][u]);
                matrix<float> s = sdiT * error;

                S += s;
            }
        }
        // ----------------------------------------------- //
        // STEP 8                                          //
        // solve the delta p                               //
        // ----------------------------------------------- //
        matrix<float> deltaP = invH * S;

        // ----------------------------------------------- //
        // STEP 9                                          //
        // and update warping matrix (ic)                  //
        // ----------------------------------------------- //
        float deltaS  = deltaP.at(0);
        float deltaB1 = deltaP.at(1);
        float deltaB2 = deltaP.at(2);

        W.at(0, 0) /= (1 + deltaS);
        W.at(1, 1)  = W.at(0, 0);
        W.at(0, 2) -= W.at(0, 0) * deltaB1;
        W.at(1, 2) -= W.at(0, 0) * deltaB2;

        cout << "----" << i << "----" << endl;
        cout << deltaP << endl;
    }
    cout << W << endl;

    return SUCCESS;
}
