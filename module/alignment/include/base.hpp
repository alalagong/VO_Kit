#ifndef _BASE_HPP_
#define _BASE_HPP_ 

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <eigen3/Eigen/Core>

#include <vector>
#include <memory>

#include <math_base.hpp>

namespace vo_kit
{

typedef std::vector<cv::Mat> ImgPyr;

class Cam{

public:
    // only one
    static Cam& getCamera()
    {
        static Cam instance;
        return instance;
    }
    static double& fx() { return getCamera().fx_; }
    static double& fy() { return getCamera().fy_; }
    static double& cx() { return getCamera().cx_; }
    static double& cy() { return getCamera().cy_; }
    static int& height() { return getCamera().height_; }
    static int& width()  { return getCamera().width_;  }
    static Eigen::Matrix3d& K() { return getCamera().K_; }

private:
    //TODO: read from files
    Cam():
    fx_(1),
    fy_(1),
    cx_(1),
    cy_(1),
    height_(480),
    width_(640)
    {
        K_ <<   fx_,    0,      cx_,
                0,      fy_,     cy_,
                0,      0,      1; 
    }
    
    Cam( Cam const&);
    void operator=(Cam const&);
        
    double fx_, fy_, cx_, cy_;
    int height_, width_;
    Eigen::Matrix3d K_;
};



class Frame
{
public:

    typedef std::shared_ptr<Frame> Ptr;
    
    size_t                      num_max_level_;     //!< Max numbers of levels of pyramid
    std::vector<cv::KeyPoint>   features;           //!< the corner of image
    static int                  frame_count_;       //!< Count the number of frames


    void createImgPyr(const cv::Mat& img);
    void setPose(const Sophus::SE3d& pose);
    Sophus::SE3d getPose() const;
    cv::Mat getImage(size_t level) const; 
    inline static Ptr create(const cv::Mat& img, const double time_stamp)
    {
        return Ptr(new Frame(img, time_stamp));
    }
    

protected:

    Frame(const cv::Mat& img, const double time_stamp);
    Frame(const Frame&) = delete;
    Frame &operator=(const Frame&) = delete;

    Sophus::SE3d     T_c_w_;             //!< Pose Transform from world to camera
    ImgPyr          img_pyr_;           //!< Image pyramid

};


} // end namespace vo_kit

#endif //_BASE_HPP_