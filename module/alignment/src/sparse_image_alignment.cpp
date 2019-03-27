#include <sparse_image_alignment.hpp>

namespace vo_kit
{

SparseImgAlign::SparseImgAlign(
    int         top_level,   
    int         bottom_level,
    size_t      num_iter_max,
    Method      method,      
    PatternType pattern,     
    double      Delta_init ) : 
    num_level_max_(top_level),
    num_level_min_(bottom_level),
    pattern_(pattern)
{
    num_iter_max_ = num_iter_max;
    method_ = method;
    Delta_init_ = Delta_init;  // for Dog-Leg
}

size_t SparseImgAlign::run(Frame::Ptr frame_ref, Frame::Ptr frame_cur)
{
    if(frame_ref->features.empty())
    {
        std::cout<<"[error]: There is nothing in features!"<<std::endl;
    }

    patch_ref_ = cv::Mat(frame_ref->features.size(), pattern_.size(), CV_32F);
    visible_ftr_.resize(frame_ref->features.size(), false);

    
    // Get the optimizing state
    Sophus::SE3d T_r_w = frame_ref->getPose();
    Sophus::SE3d T_c_w = frame_cur->getPose();
    Sophus::SE3d T_cur_ref = T_c_w * T_r_w.inverse();

    size_t all_obser;
    // Frames to aligne 
    frame_cur_ = frame_cur;
    frame_ref_ = frame_ref;
    
    for(level_cur_=num_level_max_; level_cur_ >= num_level_min_; level_cur_--)
    {
        //TODO 每次计算清空这几个Dynamic

        preCompute();  // get reference patch
 
        runOptimize(T_cur_ref); // first compute Jacobian and residual, then solve

        all_obser += num_obser_; // all pixels we used
    }
    
    // update result
    T_c_w = T_cur_ref*T_r_w;
    frame_cur->setPose(T_c_w);

    return all_obser/pattern_.size();

}

void SparseImgAlign::preCompute()
{
    cv::Mat img_ref_level = frame_ref_->getImage(level_cur_);
    size_t num_ftr_active = 0;
    size_t num_ftr = frame_ref_->features.size();
    double scale = 1.f/(1<<level_cur_); // current level
    int stride = img_ref_level.step[0]; // step

    auto iter_vis_ftr = visible_ftr_.begin();

    for(size_t i_ftr=0; i_ftr<num_ftr; ++i_ftr, ++iter_vis_ftr)
    {
        //! feature on level_cur
        const int ftr_x = frame_ref_->features[i_ftr].pt.x; // feature coordiante
        const int ftr_y = frame_ref_->features[i_ftr].pt.y;
        const float ftr_level_x = ftr_x*scale; // current level feature coordiante
        const float ftr_level_y = ftr_y*scale;
        const int ftr_level_x_i = floor(ftr_level_x); // current level feature integer coordiante
        const int ftr_level_y_i = floor(ftr_level_y);
        bool is_in_frame = true;
        //* patch pointer on head
        float* data_patch_ref = reinterpret_cast<float*>(patch_ref_.data)+i_ftr*pattern_.size();
        //* img_ref_level pointer on feature
        float* data_img_ref = reinterpret_cast<float*>(img_ref_level.data)+ftr_level_y_i*stride+ftr_level_x_i;
        int pattern_count = 0;
        for(auto iter_pattern : pattern_)
        {
            int x_pattern = iter_pattern.first;     // offset from feature 
            int y_pattern = iter_pattern.second;    
            // Is in the image of current level
            if( x_pattern + ftr_level_x_i<0 || y_pattern + ftr_level_y_i<0 || 
                x_pattern + ftr_level_x_i>img_ref_level.cols || 
                y_pattern + ftr_level_y_i>img_ref_level.rows)
            {
                is_in_frame = false;
                break;
            }
            const float* data_img_pattern = data_img_ref + y_pattern*stride + x_pattern;
            const float x_img_pattern = ftr_level_x + x_pattern;
            const float y_img_pattern = ftr_level_y + y_pattern;
            data_patch_ref[pattern_count++] = utils::interpolate_float(data_img_pattern, x_img_pattern, y_img_pattern, stride);
        }
        
        if(is_in_frame)
        {
            *iter_vis_ftr = true; 
            ++num_ftr_active;
        }
        else
            continue;
    }

    jacobian_.resize(num_ftr_active, Eigen::NoChange);
    residual_.resize(num_ftr_active, Eigen::NoChange);
}



} // end vo_kit