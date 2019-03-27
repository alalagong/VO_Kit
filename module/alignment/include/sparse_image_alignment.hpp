#ifndef _SPARSE_IMAGE_ALIGNMENT_HPP_
#define _SPARSE_IMAGE_ALIGNMENT_HPP_

#include <base.hpp>
#include <optimizer_base.h>


namespace vo_kit
{
//* Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlign : public Optimizer<6, Sophus::SE3d>
{
    typedef std::vector<std::pair<int, int> > PatternType;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SparseImgAlign(
        int         top_level,      //!< start level numbers of alignment       
        int         bottom_level,   //!< end level numbers of alignment
        size_t      num_iter_max,   //!< max iteration from father class
        Method      method,         //!< descent method GN/LM/DL from father class
        PatternType pattern,        //!< patch pattern of alignment
        double      Delta_init     //!< just for Dog-Leg method
    );

    size_t run(Frame::Ptr frame_ref, Frame::Ptr frame_cur);

private:

    int                 level_cur_;     //!< current level for optimizing
    int                 num_level_min_; //!< bottom level
    int                 num_level_max_; //!< top level
    cv::Mat             patch_ref_;     //!< the reference patch rows=featurenum cols=patchsize 
    PatternType         pattern_;       //!< pattern of patch, should put feature(center) to (0,0)
    std::vector<bool>   visible_ftr_;   //!< the visibility of feature projected from reference frame  
    Frame::Ptr          frame_ref_;     //!< frame of reference 
    Frame::Ptr          frame_cur_;     //!< frame of current

    void  preCompute();
    virtual void computeResidual(const Sophus::SE3d& state);
    virtual void computeJacobian(const Sophus::SE3d& state);
    virtual void update(const Sophus::SE3d& old_state, Sophus::SE3d& new_state);
    virtual bool solve();
};


} // end vo_kit

#endif