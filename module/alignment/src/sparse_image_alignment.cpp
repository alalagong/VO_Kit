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

    patch_ref_ = cv::Mat::zeros(frame_ref->features.size(), pattern_.size(), CV_32F);
    patch_dI_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    visible_ftr_.resize(frame_ref->features.size(), false);
    jacobian_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    residual_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    
    num_ftr_active_ = 0;
    
    // Get the optimizing state
    Sophus::SE3d T_r_w = frame_ref->getPose();
    Sophus::SE3d T_c_w = frame_cur->getPose();
    Sophus::SE3d T_cur_ref = T_c_w * T_r_w.inverse();

    // size_t all_obser;
    // Frames to aligne 
    frame_cur_ = frame_cur;
    frame_ref_ = frame_ref;
    
    for(level_cur_=num_level_max_; level_cur_ >= num_level_min_; level_cur_--)
    {
        preCompute();  // get reference patch
 
        runOptimize(T_cur_ref); // first compute Jacobian and residual, then solve

        // all_obser += num_ftr_active_; // all pixels we used
    }
    
    // update result
    T_c_w = T_cur_ref*T_r_w;
    frame_cur->setPose(T_c_w);

    return num_ftr_active_;

}

void SparseImgAlign::preCompute()
{
    cv::Mat img_ref_pyr = frame_ref_->getImage(level_cur_);
    patch_dI_.setZero();
    // size_t num_ftr_active = 0;
    size_t num_ftr = frame_ref_->features.size();
    const double scale = 1.f/(1<<level_cur_); // current level
    int stride = img_ref_pyr.step[0]; // step

    auto iter_vis_ftr = visible_ftr_.begin();

    for(size_t i_ftr=0; i_ftr<num_ftr; ++i_ftr, ++iter_vis_ftr)
    {
        //! feature on level_cur
        const int ftr_x = frame_ref_->features[i_ftr].pt.x; // feature coordiante
        const int ftr_y = frame_ref_->features[i_ftr].pt.y;
        const float ftr_pyr_x = ftr_x*scale; // current level feature coordiante
        const float ftr_pyr_y = ftr_y*scale;
        // const int ftr_pyr_x_i = floor(ftr_pyr_x); // current level feature integer coordiante
        // const int ftr_pyr_y_i = floor(ftr_pyr_y);
        bool is_in_frame = true;
        //* patch pointer on head
        float* data_patch_ref = reinterpret_cast<float*>(patch_ref_.data)+i_ftr*pattern_.size();
        //* img_ref_pyr pointer on feature
        // float* data_img_ref = reinterpret_cast<float*> (img_ref_pyr.data)+ftr_pyr_y_i*stride+ftr_pyr_x_i;
        int pattern_count = 0;

        for(auto iter_pattern : pattern_)
        {
            int x_pattern = iter_pattern.first;     // offset from feature 
            int y_pattern = iter_pattern.second;    
            // Is in the image of current level
            if( x_pattern + ftr_pyr_x < 1 || y_pattern + ftr_pyr_y < 1 || 
                x_pattern + ftr_pyr_x > img_ref_pyr.cols - 2 || 
                y_pattern + ftr_pyr_y > img_ref_pyr.rows - 2 )
            {
                is_in_frame = false;
                break;
            }
            
            // reference patch
            // const float* data_img_pattern = data_img_ref + y_pattern*stride + x_pattern;
            const float x_img_pattern = ftr_pyr_x + x_pattern;
            const float y_img_pattern = ftr_pyr_y + y_pattern;
            data_patch_ref[pattern_count] = utils::interpolate_float(reinterpret_cast<float*>(img_ref_pyr.data), x_img_pattern, y_img_pattern, stride);
            
            // derive patch
            float dx = 0.5*(utils::interpolate_float(reinterpret_cast<float*>(img_ref_pyr.data), x_img_pattern+1, y_img_pattern, stride) - 
                            utils::interpolate_float(reinterpret_cast<float*>(img_ref_pyr.data), x_img_pattern-1, y_img_pattern, stride));
            float dy = 0.5*(utils::interpolate_float(reinterpret_cast<float*>(img_ref_pyr.data), x_img_pattern, y_img_pattern+1, stride) - 
                            utils::interpolate_float(reinterpret_cast<float*>(img_ref_pyr.data), x_img_pattern, y_img_pattern-1, stride));
            patch_dI_.row(i_ftr*pattern_.size()+pattern_count) = Eigen::Vector2d(dx, dy);
            ++pattern_count;
        }
        
        if(is_in_frame)
        {
            *iter_vis_ftr = true; 
            // ++num_ftr_active;
        }
    }

    // jacobian_.resize(num_ftr_active, Eigen::NoChange);
    // residual_.resize(num_ftr_active, Eigen::NoChange);
}



//! NOTE myself: need to make sure residual match Jacobians
void SparseImgAlign::computeResidual(const Sophus::SE3d& state)
{
    residual_.setZero();
    jacobian_.setZero();
    Sophus::SE3d T_cur_ref = state;
    cv::Mat img_cur_pyr = frame_cur_->getImage(level_cur_);
    size_t num_ftr = frame_ref_->features.size();
    int num_pattern = pattern_.size();
    int stride = img_cur_pyr.step[0];
    const double scale = 1.f/(1<<level_cur_);
    auto iter_vis_ftr = visible_ftr_.begin();

    for(size_t i_ftr=0; i_ftr<num_ftr; ++i_ftr, ++iter_vis_ftr)
    {
        if(!*iter_vis_ftr)
            continue;
        
        // get point in reference
        const int ftr_ref_x = frame_ref_->features[i_ftr].pt.x;
        const int ftr_ref_y = frame_ref_->features[i_ftr].pt.y;
        const double ftr_depth_ref = frame_ref_->getDepth(ftr_ref_x, ftr_ref_y);
        // project to current 
        Eigen::Vector3d point_ref(Cam::pixel2unitPlane(ftr_ref_x, ftr_ref_y)*ftr_depth_ref);
        Eigen::Vector3d point_cur(T_cur_ref*point_ref);
        Eigen::Vector2d ftr_cur = Cam::project(point_cur)*scale;

        const float ftr_cur_x = ftr_cur.x();
        const float ftr_cur_y = ftr_cur.y();
        // const int ftr_cur_x_i = floor(ftr_cur_x);
        // const int ftr_cur_y_i = floor(ftr_cur_y);

        bool is_in_frame = true;
        Eigen::VectorXd residual_pattern = Eigen::VectorXd::Zero(num_pattern);
        size_t pattern_count = 0; 

        float* data_patch_ref = reinterpret_cast<float*>(patch_ref_.data) + i_ftr*num_pattern;

        for(auto iter_pattern : pattern_)
        {
            int x_pattern = iter_pattern.first;     // offset from feature 
            int y_pattern = iter_pattern.second;    
            // Is in the image of current level
            if( x_pattern + ftr_cur_x < 1 || y_pattern + ftr_cur_y < 1 || 
                x_pattern + ftr_cur_x > img_cur_pyr.cols - 2 || 
                y_pattern + ftr_cur_y > img_cur_pyr.rows - 2 )
            {
                is_in_frame = false;
                break;
            }

            const float x_img_pattern = ftr_cur_x + x_pattern;
            const float y_img_pattern = ftr_cur_y + y_pattern;
            float pattern_value = utils::interpolate_float(reinterpret_cast<float*>(img_cur_pyr.data), x_img_pattern, y_img_pattern, stride);
            // res = T-I
            residual_pattern[pattern_count] = data_patch_ref[pattern_count] - pattern_value;
            ++pattern_count;
        }

        if(is_in_frame)
        {
            residual_.segment(i_ftr*num_pattern, num_pattern) = residual_pattern;
            num_ftr_active_++;
        }
        else
        {
            *iter_vis_ftr = false;
            continue;
        }

        // calculate jacobians
        {
            Eigen::Matrix<double, 2, 6> jacob_xyz2uv;
            const double x = point_ref[0];
            const double y = point_ref[1];
            const double z_inv = 1./point_ref[2];
            const double z_inv_2 = z_inv*z_inv;
            jacob_xyz2uv(0,0) = -z_inv;                         // -1/z
            jacob_xyz2uv(0,1) = 0.0;                            // 0
            jacob_xyz2uv(0,2) = x*z_inv_2;                      // x/z^2
            jacob_xyz2uv(0,3) = y*jacob_xyz2uv(0,2);            // x*y/z^2
            jacob_xyz2uv(0,4) = -(1.0 + x*jacob_xyz2uv(0,2));   // -(1.0 + x^2/z^2)
            jacob_xyz2uv(0,5) = y*z_inv;                        // y/z
            jacob_xyz2uv(1,0) = 0.0;                            // 0
            jacob_xyz2uv(1,1) = -z_inv;                         // -1/z
            jacob_xyz2uv(1,2) = y*z_inv_2;                      // y/z^2
            jacob_xyz2uv(1,3) = 1.0 + y*jacob_xyz2uv(1,2);      // 1.0 + y^2/z^2
            jacob_xyz2uv(1,4) = -jacob_xyz2uv(0,3);             // -x*y/z^2
            jacob_xyz2uv(1,5) = -x*z_inv;                       // x/z

            jacob_xyz2uv.row(0) *= Cam::fx();
            jacob_xyz2uv.row(1) *= Cam::fy();

            jacobian_.block(num_pattern*i_ftr, 0, num_pattern, 6) = 
                    patch_dI_.block(num_pattern*i_ftr, 0, num_pattern, 2) * jacob_xyz2uv;
        }
    }

}


//bug: 当初optimizer写的太死了，逆向组合不用每次计算jacobian....
void SparseImgAlign::computeJacobian(const Sophus::SE3d& state)
{
    //! 内部每次还得计算hessian (；′⌒`)
}


bool SparseImgAlign::solve()
{
    delta_x_ = hessian_.ldlt().solve(jres_);
    if((double)std::isnan(delta_x_[0]))
        return false;
    return true;
}

void SparseImgAlign::update(const Sophus::SE3d& old_state, Sophus::SE3d& new_state)
{
    new_state = old_state*Sophus::SE3d::exp(delta_x_);
}


} // end vo_kit