#include <base.hpp>


namespace vo_kit
{

int Frame::frame_count_ = 0; 

Frame::Frame(const cv::Mat& img, const cv::Mat& depth, const double time_stamp):
    num_max_level_(4)
{   
    assert(!img.empty() && !depth.empty());
    assert(img.type() == CV_8UC1);
    assert(depth.type() == CV_32FC1);

    T_c_w_ = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

    // build image pyramid
    createImgPyr(img);
    img_depth_ = depth;
}

cv::Mat Frame::getImage(size_t level) const
{
    assert(level <= num_max_level_ && level > 0);
    return img_pyr_[level];
}

void Frame::setPose(const Sophus::SE3d& pose)
{
    T_c_w_ = pose;
}

Sophus::SE3d Frame::getPose() const
{
    return T_c_w_;
}

float Frame::getDepth(int u, int v) const
{
    return img_depth_.at<float>(v, u);
}

void Frame::createImgPyr(const cv::Mat& img)
{
    img_pyr_.resize(num_max_level_);
    img_pyr_[0] = img;

    for(size_t i=1; i<num_max_level_; ++i)
    {
        img_pyr_[i] = cv::Mat(img_pyr_[i-1].rows/2, img_pyr_[i-1].cols/2, CV_8U);
        utils::halfSample(img_pyr_[i-1],img_pyr_[i]);
    }

}

} //end vo_kit