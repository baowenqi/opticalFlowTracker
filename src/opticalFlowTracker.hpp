#pragma once

class ofTracker
{
    public:

    struct box
    {
        float   x;
        float   y;
        float   w;
        float   h;
        box(int32_t i_x = 0, int32_t i_y = 0, int32_t i_w = 0, int32_t i_h = 0):
        x(i_x), y(i_y), w(i_w), h(i_h) {};
       ~box() {};
    };
    
    struct image
    {
        int32_t   w;
        int32_t   h;
        float*    data;
        image(int32_t i_w, int32_t i_h) : w(i_w), h(i_h) { data = new float[w * h]; };
       ~image() { if(data != nullptr) delete []data; };
    };

    float*        outW;
    box           outBox;
    typedef enum  e_status {SUCCESS, ERROR} status_t;

                  ofTracker(int32_t i_imgWidth, int32_t i_imgHeight);
                 ~ofTracker();
    status_t      track();

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
    box*          m_boxPyd0[m_numPyramid];
    box*          m_boxPyd1[m_numPyramid];

    status_t      f_convolution(float* src, float* dst, float* knl, int32_t kw, int32_t kh, int32_t w, int32_t h);
    float         f_sample(image& img, float x, float y);

    status_t      f_buildPyramid(float* frame, image* (&pyramid)[m_numPyramid]);
    status_t      f_align(box& tmpBox);
};

