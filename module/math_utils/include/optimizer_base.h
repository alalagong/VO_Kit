#ifndef _OPTIMIZER_BASE_H_
#define _OPTIMIZER_BASE_H_

#include <iostream>
#include <vector>
#include <math_base.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <iomanip>

namespace vo_kit
{
using namespace std;
using namespace Eigen;

/*===============================
 * @ class: least-square Minimization
 * 
 * @ param:   D : dimension of the states
 *            T : type of the state, e.g. SE2, SE3
 * 
 * @ note:   TODO: Prior and robust cost function need to consider
 ================================*/
template <int D, typename T>
class Optimizer
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef T StateType;
    enum Method{GaussNewton, LevenbergMarquardt, DogLeg};

    // base control variable
    size_t      num_iter_max_;          //!< max numbers of iteration
    size_t      num_iter_;              //!< current number of iteration
    double      epsilon_;               //!< convergence condition
    bool        stop_;                  //!< stop flag
    Method      method_;                //!< method of descent
    size_t      num_obser_;             //!< numbers of measurement

    Optimizer() : 
        num_iter_max_(15),
        num_iter_(0),
        // num_trials_max_(5),
        // num_trials_(0),
        epsilon_(0.00000001),
        stop_(false),
        method_(GaussNewton),
        num_obser_(0),
        converged_(false),
        lambda_factor_(2),
        lambda_max_(1e5),   // from GTSAM3.2.1
        lambda_min_(0.0),
        min_fidelity_(1e-3),
        Delta_init_(1.5),
        Delta_(Delta_init_)
        { }

    // run optimize
    bool runOptimize(StateType& state);
    // get residual (will waste time ?)
    double getChi2(bool is_linear)
    {
        Matrix<double, Dynamic, 1> residual_linear;
        residual_linear = residual_ - jacobian_*delta_x_;
        if(is_linear)
            return residual_linear.transpose()*residual_linear;
        else
            return residual_.transpose()*residual_;
    };
    // Reset
    void reset()
    {
        Delta_=Delta_init_;
        rho_=-1;
        chi2_=0;
        lambda_factor_=2;
        lambda_=-1;
        num_obser_=0;
        // num_trials_=0;        
        num_iter_ = 0;
        stop_=false;
        converged_=false;
        beta_ = 0;
    };
    // get infomation Matrix
    Matrix<double, D, D>& getInformationMatrix() const
    {
        return hessian_;
    };
    
protected:

    // base optimize variable
    Matrix<double, Dynamic, D>  jacobian_;      //!< Jacobians of state 
    Matrix<double, Dynamic, 1>  residual_;      //!< residuals between prediction and observation
    Matrix<double, D, D>        hessian_;       //!< Hessians Matrix
    Matrix<double, D, 1>        jres_;          //!< Jacobians * residuals
    Matrix<double, D, 1>        delta_x_;       //!< increment
    MatrixXd                    cov_;           //!< covariance of observation
    double                      chi2_;          //!< Sum of squares of random variables obeying Gaussian distribution
    bool                        converged_;     //!< whether converge to enough small
    // for LM   
    double                      lambda_;        //!< multiplier
    double                      lambda_factor_; //!< expand factor of multiplier
    double                      lambda_max_;    //!< upper bound of lambda
    double                      lambda_min_;    //!< lower bound of lambda
    // size_t                      num_trials_max_;//!< max numbers of trials of slove after failed
    // size_t                      num_trials_;    //!< current numbers of trials
    double                      rho_;           //!< modelFidelity to adjust trust region
    double                      min_fidelity_;  //!< Lower bound for the modelFidelity to accept the result of an LM iteration
    // for Dog-Leg (this is a big hole)
    Matrix<double, D, 1>        delta_GN_x_;    //!< step from GaussNewton
    Matrix<double, D, 1>        delta_SD_x_;    //!< step from SteepestDescent consists of direction and length
    double                      beta_;          //!< when GN exceed trust region and SD not exceed it, need beta to make delta_x = region
    double                      Delta_init_;    //!< initial radius of trust region
    double                      Delta_;         //!< radius of trust region 
 

    // optimize in method of GaussNetow
    void optimizeGaussNetow(StateType& state);
    // optimize in method of LevenbergMarquardt
    void optimizeLevenbergMarquardt(StateType& state);
    // increase lambda for LM
    void increaseLambda();
    // decrease lambda for LM
    void decreaseLambda(double& rho);
    // optimize in method of Dog-Leg
    void optimizeDogLeg(StateType& state);
    // compute descent step by method GN for Dog-Leg
    void computeDeltaGN(StateType& state);
    // compute descent step by method SD for Dog-Leg
    void computeDeltaSD();
    // compute descent step by method of Dog-Leg
    void computeDeltaDL();
    // compute beta for Dog-Leg
    void computeBeta();
    // calculate residual
    virtual void computeResidual(const StateType& state) = 0;
    // calculate Jacobians
    virtual void computeJacobian(const StateType& state) = 0;
    // update states
    virtual void update(const StateType& old_state, StateType& new_state) = 0;
    // solve method
    virtual bool solve() = 0;
    
    // start compute
    virtual void startIteration();
    // finish compute
    virtual void finishIteration();
    
    
};
} //end namespace vo_kit

// have to do this
#include "optimizer_base.hpp"
#endif