#ifndef _MATH_BASE_HPP_
#define _MATH_BASE_HPP_
 
#include <vector>
#include <eigen3/Eigen/Core>

namespace vo_kit
{


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

} //end namespace

#endif // _MATH_BASE_HPP_