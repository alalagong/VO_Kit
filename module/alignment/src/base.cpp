#include <base.hpp>


namespace vo_kit
{

int Frame::frame_count_ = 0; 

Frame::Frame(const cv::Mat& img, const cv::Mat& depth, const size_t max_pyr):
    num_max_level_(max_pyr)
{   
    assert(!img.empty() && !depth.empty());
    assert(img.type() == CV_8UC1);
    assert(depth.type() == CV_32FC1);

    T_c_w_ = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    ++frame_count_;
    // build image pyramid
    createImgPyr(img);
    img_depth_ = depth;
}


void Frame::setPose(const Sophus::SE3d& pose)
{
    T_c_w_ = pose;
}


void Frame::createImgPyr(const cv::Mat& img)
{
    img_pyr_.resize(num_max_level_+1);
    img_pyr_[0] = img;

    for(size_t i=1; i<=num_max_level_; ++i)
    {
        img_pyr_[i] = cv::Mat(img_pyr_[i-1].rows/2, img_pyr_[i-1].cols/2, CV_8U);
        utils::halfSample(img_pyr_[i-1],img_pyr_[i]);
    }

}

} //end vo_kit