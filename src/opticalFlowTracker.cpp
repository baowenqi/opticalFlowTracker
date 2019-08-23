#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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

        m_boxPyd0[i] = new box;
        m_boxPyd1[i] = new box;
    }
}

ofTracker::~ofTracker()
{
    if(outW != nullptr) delete []outW;

    for(int i = (m_numPyramid - 1); i >= 0; i--)
    {
        if(m_imgPyd0[i] != nullptr) delete m_imgPyd0[i];
        if(m_imgPyd1[i] != nullptr) delete m_imgPyd1[i];
        if(m_boxPyd0[i] != nullptr) delete m_boxPyd0[i];
        if(m_boxPyd1[i] != nullptr) delete m_boxPyd1[i];
    }
}

ofTracker::status_t ofTracker::f_convolution
(
    float*    src,
    float*    dst,
    float*    knl,
    int32_t   kw,
    int32_t   kh,
    int32_t   w,
    int32_t   h
)
{
    int32_t kwRad = (kw >> 1);
    int32_t khRad = (kh >> 1);
    int32_t wPad = w + kwRad * 2;
    int32_t hPad = h + khRad * 2;

    // ---------------------------------------------- //
    // pad the boundry of the src image with 0s       //
    // ---------------------------------------------- //
    float* srcPad = new float[wPad * hPad];
    memset(srcPad, 0, wPad * hPad * sizeof(float));
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            int srcAddr = y * w + x;
            int padAddr = (y + khRad) * wPad + (x + kwRad);
            *(srcPad + padAddr) = *(src + srcAddr);
        }
    }

    // ---------------------------------------------- //
    // convolve the srcPad with knl                   //
    // ---------------------------------------------- //
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            float result = 0.0f;
            for(int v = 0; v < kh; v++)
            {
                for(int u = 0; u < kw; u++)
                {
                    int knlAddr = v * kw + u;
                    int srcAddr = (y + v) * wPad + (x + u);
                    
                    float knlData = *(knl + knlAddr);
                    float srcData = *(srcPad + srcAddr);
                    result += srcData * knlData;
                }
            }
            int dstAddr = y * w + x;
            *(dst + dstAddr) = result;
        }
    }

    delete []srcPad;
    return SUCCESS;
}

float ofTracker::f_sample
(
    image&    img,
    float     x,
    float     y
)
{
    int ix = static_cast<int>(floor(x));
    int iy = static_cast<int>(floor(y));

    float d0 = *(img.data + (iy + 0) * img.w + (ix + 0));
    float d1 = *(img.data + (iy + 0) * img.w + (ix + 1));
    float d2 = *(img.data + (iy + 1) * img.w + (ix + 0));
    float d3 = *(img.data + (iy + 1) * img.w + (ix + 1));

    float a = x - ix;
    float b = y - iy;

    return (d0 * (1 - a) + d1 * a) * (1 - b) + (d2 * (1 - a) + d3 * a) * b;
}

ofTracker::status_t ofTracker::f_buildPyramid
(
    float* frame, 
    image* (&pyramid)[m_numPyramid]
)
{
    // ---------------------------------------------- //
    // gaussian filter kernel                         //
    // ---------------------------------------------- //
    float knl[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};

    // ---------------------------------------------- //
    // the bottom level is just a copy of input frame //
    // the upper-4 level is down-scaled by 2 each     //
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

            f_convolution(pyramid[l+1]->data, gaussianImg.data, knl, 3, 3, pyramid[l+1]->w, pyramid[l+1]->h);

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

