#include <optimizer_base.h>

namespace test
{

// vo_kit optimizer on linear least-square
// z = x^2 + cosy    z=10
// z= x^2 + y^2 z=0
class OptimizerTest : public vo_kit::Optimizer<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    OptimizerTest(int num_iter_max, Method method)
    {
        num_iter_max_ = num_iter_max;
        method_ = method;
    }

    virtual void computeResidual(const Eigen::Vector2d& state)
    {   
        residual_.resize(1,1);
        residual_ << 10 - state[0]*state[0] - cos(state[1]);
        // residual_ << - state[0]*state[0] - state[1]*state[1];
    }
    virtual void computeJacobian(const Eigen::Vector2d& state)
    {
        jacobian_.resize(1,2);
        jacobian_<< 2*state[0], -sin(state[1]);
        // jacobian_<< 2*state[0], 2*state[1];
    }
    virtual void update(const Eigen::Vector2d& old_state, Eigen::Vector2d& new_state)
    {   
        new_state = old_state+ delta_x_;
    }
    virtual bool solve()
    {
        delta_x_ = hessian_.ldlt().solve(jres_);
        if((double)std::isnan(delta_x_[0]))
            return false;
        return true;
    }
    
    };

}

int main()
{
    Eigen::Vector2d init;
    init << 100 , 100;
    test::OptimizerTest test1(100, test::OptimizerTest::GaussNewton);
    test1.runOptimize(init);
    std::cout<< init <<std::endl;

    init<<11,69;
    test1.method_ = test::OptimizerTest::LevenbergMarquardt;
    test1.runOptimize(init);
    std::cout<< init <<std::endl;
        
    init<<100,100;
    test1.method_ = test::OptimizerTest::DogLeg;
    test1.runOptimize(init);
    std::cout<< init <<std::endl;
    return 1;
}
