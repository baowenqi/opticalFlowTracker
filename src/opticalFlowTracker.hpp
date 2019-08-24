#pragma once
#include <iostream>

using namespace std;

class ofTracker
{
    public:

    struct box
    {
        int32_t   x;
        int32_t   y;
        int32_t   w;
        int32_t   h;

        box(int32_t i_x = 0, int32_t i_y = 0, int32_t i_w = 0, int32_t i_h = 0):
        x(i_x), y(i_y), w(i_w), h(i_h) {};

       ~box() {};

        box& operator=(const box& rhs)
        {
            this->x = rhs.x;
            this->y = rhs.y;
            this->w = rhs.w;
            this->h = rhs.h;
            return *this;
        }

        box& operator*(const float scale)
        {
            this->x = static_cast<int32_t>(static_cast<float>(this->x) * scale);
            this->y = static_cast<int32_t>(static_cast<float>(this->y) * scale);
            this->w = static_cast<int32_t>(static_cast<float>(this->w) * scale);
            this->h = static_cast<int32_t>(static_cast<float>(this->h) * scale);
            return *this;
        }

        friend ostream& operator<<(ostream& os, const box& ctx)
        {
            os << ctx.x << "," << ctx.y << ", " << ctx.w << "," << ctx.h;
            return os;
        }
    };
    
    struct image
    {
        int32_t   w;
        int32_t   h;
        int32_t   selfCreated;
        float*    data;
        image(int32_t i_w, int32_t i_h, float* i_data = nullptr) : w(i_w), h(i_h), data(i_data)
        {
            selfCreated = 0;
            if(data == nullptr)
            {
                data = new float[w * h];
                selfCreated = 1;
            }
        };
       ~image() { if(data != nullptr && selfCreated) delete []data; };
    };

    float*        outW;
    box           outBox;
    typedef enum  e_status {SUCCESS, ERROR} status_t;

                  ofTracker(int32_t i_imgWidth, int32_t i_imgHeight);
                 ~ofTracker();
    status_t      track(box& inputBox);

    // private:
    static const     int32_t m_numPyramid = 4;
    static const     int32_t m_maxItr     = 100;
    static constexpr float   m_cvgEpsilon = 1e-6;
    static const     int32_t m_dof        = 3;

    int32_t       m_imgWidth;
    int32_t       m_imgHeight;
    float*        m_frame0;
    float*        m_frame1;
    image*        m_imgPyd0[m_numPyramid];
    image*        m_imgPyd1[m_numPyramid];
    box*          m_boxPyd[m_numPyramid];

    status_t      f_convolution(const image& src, const image& knl, image& dst);
    float         f_sample(image& img, float x, float y);

    status_t      f_buildPyramid(float* frame, image* (&pyramid)[m_numPyramid]);
    status_t      f_align(image& tmpImg, image& tgtImg, box& tmpBox, box& tgtBox);
};

