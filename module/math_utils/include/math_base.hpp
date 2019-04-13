#ifndef _MATH_BASE_HPP_
#define _MATH_BASE_HPP_
 

#include <eigen3/Eigen/Core>
#include <opencv2/core.hpp>
#include <emmintrin.h>
// #include <string>
// #include <string.h>
#include <vector>
#include <cassert>
#include <cstdlib>

namespace vo_kit
{

namespace utils
{

typedef unsigned char uint8_t;

inline double maxFabs(const Eigen::VectorXd& v)
{
    double max = -1.;
    // turn Eigen::vectorXd to std::vector
    std::vector<double> value(v.data(), v.data()+v.rows()*v.cols());
    // find max
    std::for_each(value.begin(), value.end(), [&](double& val){
        max = fabs(val) > max ? fabs(val): max;
    });
    return max;
}

// from rpg_vikit vision.cpp
 inline bool is_aligned16(const void* ptr)
  {
    return ((reinterpret_cast<size_t>(ptr)) & 0xF) == 0;
  }


inline void halfSampleSSE2(const unsigned char* in, unsigned char* out, int w, int h)
{
  const unsigned long long mask[2] = {0x00FF00FF00FF00FFull, 0x00FF00FF00FF00FFull};
  const unsigned char* nextRow = in + w;
  __m128i m = _mm_loadu_si128((const __m128i*)mask);
  int sw = w >> 4;
  int sh = h >> 1;
  for (int i=0; i<sh; i++)
  {
    for (int j=0; j<sw; j++)
    {
      __m128i here = _mm_load_si128((const __m128i*)in);
      __m128i next = _mm_load_si128((const __m128i*)nextRow);
      here = _mm_avg_epu8(here,next);
      next = _mm_and_si128(_mm_srli_si128(here,1), m);
      here = _mm_and_si128(here,m);
      here = _mm_avg_epu16(here, next);
      _mm_storel_epi64((__m128i*)out, _mm_packus_epi16(here,here));
      in += 16;
      nextRow += 16;
      out += 8;
    }
    in += w;
    nextRow += w;
  }
}

inline void halfSample(const cv::Mat& in, cv::Mat& out)
{
  assert( in.rows/2==out.rows && in.cols/2==out.cols );
  assert( in.type()==CV_8U && out.type()==CV_8U );

  if(is_aligned16(in.data) &&is_aligned16(out.data) && ((in.cols % 16) == 0))
  {
    halfSampleSSE2(in.data, out.data, in.cols, in.rows);
    return;
  }
    // four pixel merge to one
  const int stride = in.step.p[0];
  uint8_t* top = (uint8_t*) in.data;
  uint8_t* bottom = top + stride;
  uint8_t* end = top + stride*in.rows;
  const int out_width = out.cols;
  uint8_t* p = (uint8_t*) out.data;
  while (bottom < end)
  {
    for (int j=0; j<out_width; j++)
    {
      *p = static_cast<uint8_t>( (uint16_t (top[0]) + top[1] + bottom[0] + bottom[1])/4 );
      p++;
      top += 2;
      bottom += 2;
    }
    top += stride;
    bottom += stride;
  }

}

/******************
 * 坐标系  -------> x
 *         |
 *         |
 *         y
 ******************/
//! the data should be the origin, or it may appear memory leak
inline float interpolate_float(const float* data, const float x, const float y, const int stride)
{
  const int x_i = floor(x);
  const int y_i = floor(y);
  const float subpix_x = x - x_i;
  const float subpix_y = y - y_i;
  const float w_tl = (1.f-subpix_x) * (1.f-subpix_y);
  const float w_tr = (subpix_x) * (1.f-subpix_y);
  const float w_bl = (1.f-subpix_x) * (subpix_y);
  const float w_br = subpix_x * subpix_y;

  const float* data_cur = data + y_i*stride + x_i; 

  return w_tl*data_cur[0] + w_tr*data_cur[1] + w_bl*data_cur[stride] + w_br*data_cur[stride+1];
}

}
} //end namespace

#endif // _MATH_BASE_HPP_