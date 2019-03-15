#include <optimizer_base.hpp>

//namespace test
namespace vo_kit
{

// using namespace vo_kit;
// vo_kit optimizer on linear least-square
// z = x^2 + cosy    z=10
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
        residual_ << state[0]*state[0]+cos(state[1]) - 10;
    }
    virtual void computeJacobian(const Eigen::Vector2d& state)
    {
        jacobian_.resize(1,2);
        jacobian_<< 2*state[0], -sin(state[1]);
    }
    virtual void update(const Eigen::Vector2d& old_state, Eigen::Vector2d& new_state)
    {   
        new_state= old_state+ delta_x_;
    }
    virtual bool solve()
    {
        delta_x_ = hessian_.inverse()*jacobian_.transpose()*residual_;
        if(std::isnan(delta_x_[0]))
            return false;
        return true;
    }
    void run(Eigen::Vector2d init)
    {
        runOptimize(init);
    }
    
    
    };

}

int main()
{
    Eigen::Vector2d init;
    init<< 1 , 1;
    //test::OptimizerTest test1(20, test::OptimizerTest::GaussNewton);
    vo_kit::OptimizerTest test1(20, vo_kit::OptimizerTest::GaussNewton);
    //test1.run(init);
    //test1.runOptimize(init);
    test1.test();
    std::cout<< init <<std::endl;
    return 1;
}
