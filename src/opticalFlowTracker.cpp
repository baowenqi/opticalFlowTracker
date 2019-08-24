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

        m_boxPyd[i] = new box;
    }
}

ofTracker::~ofTracker()
{
    if(outW != nullptr) delete []outW;

    for(int i = (m_numPyramid - 1); i >= 0; i--)
    {
        if(m_imgPyd0[i] != nullptr) delete m_imgPyd0[i];
        if(m_imgPyd1[i] != nullptr) delete m_imgPyd1[i];
        if(m_boxPyd[i] != nullptr) delete m_boxPyd[i];
    }
}

ofTracker::status_t ofTracker::f_convolution
(
    const image&    src,
    const image&    knl,
          image&    dst
)
{
    assert(src.w == dst.w && src.h == dst.h);

    int32_t kwRad = (knl.w >> 1);
    int32_t khRad = (knl.h >> 1);
    int32_t wPad = src.w + kwRad * 2;
    int32_t hPad = src.h + khRad * 2;

    // ---------------------------------------------- //
    // pad the boundry of the src image with 0s       //
    // ---------------------------------------------- //
    float* srcPad = new float[wPad * hPad];
    memset(srcPad, 0, wPad * hPad * sizeof(float));
    for(int y = 0; y < src.h; y++)
    {
        for(int x = 0; x < src.w; x++)
        {
            int srcAddr = y * src.w + x;
            int padAddr = (y + khRad) * wPad + (x + kwRad);
            *(srcPad + padAddr) = *(src.data + srcAddr);
        }
    }

    // ---------------------------------------------- //
    // convolve the srcPad with knl                   //
    // ---------------------------------------------- //
    for(int y = 0; y < dst.h; y++)
    {
        for(int x = 0; x < dst.w; x++)
        {
            float result = 0.0f;
            for(int v = 0; v < knl.h; v++)
            {
                for(int u = 0; u < knl.w; u++)
                {
                    int knlAddr = v * knl.w + u;
                    int srcAddr = (y + v) * wPad + (x + u);
                    
                    float knlData = *(knl.data + knlAddr);
                    float srcData = *(srcPad + srcAddr);
                    result += srcData * knlData;
                }
            }
            int dstAddr = y * dst.w + x;
            *(dst.data + dstAddr) = result;
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

    float result = 0.0f;

    if(ix >= 0 && ix < img.w - 1 && iy >= 0 && iy < img.h - 1)
    {
        float d0 = *(img.data + (iy + 0) * img.w + (ix + 0));
        float d1 = *(img.data + (iy + 0) * img.w + (ix + 1));
        float d2 = *(img.data + (iy + 1) * img.w + (ix + 0));
        float d3 = *(img.data + (iy + 1) * img.w + (ix + 1));

        float a = x - ix;
        float b = y - iy;

        result = (d0 * (1 - a) + d1 * a) * (1 - b) + (d2 * (1 - a) + d3 * a) * b;
    }

    return result;
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
    image&    tmpImg,
    image&    tgtImg,
    box&      tmpBox,
    box&      tgtBox
)
{
    // ----------------------------------------------- //
    // initial the warping matrix                      //
    // we use scaling + translation model:             //
    // 1+s, 0,   b1,                                   //
    // 0,   1+s, b2                                    //
    // ----------------------------------------------- //
    float W[2][3] = {{1, 0, 0}, {0, 1, 0}};

    // ----------------------------------------------- //
    // initial the hessian matrix                      //
    // ----------------------------------------------- //
    float H[3][3] = {0};

    // ----------------------------------------------- //
    // STEP 3                                          //
    // compute the gradient of template image          //
    // we use sobel filter here                        //
    // ----------------------------------------------- //
    float sobelX[9] = {-1,  0,  1, -2, 0, 2, -1, 0, 1};
    float sobelY[9] = {-1, -2, -1,  0, 0, 0,  1, 2, 1};
    image sobelImgX(3, 3, sobelX);
    image sobelImgY(3, 3, sobelY);

    image tmpImgGx(tmpImg.w, tmpImg.h);
    image tmpImgGy(tmpImg.w, tmpImg.h);

    f_convolution(tmpImg, tmpImgGx, sobelImgX);
    f_convolution(tmpImg, tmpImgGy, sobelImgY);

    return SUCCESS;
}
