// #ifndef _OPTIMIZER_BASE_HPP_
// #define _OPTIMIZER_BASE_HPP_

#include <optimizer_base.hpp>

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

template<int D, typename T>
bool vo_kit::Optimizer<D, T>::runOptimize(StateType& state)
{   
    if(method_ == GaussNewton)
        optimizeGaussNetow(state);
    else if(method_ == LevenbergMarquardt)
        optimizeLevenbergMarquardt(state);
    else if(method_ == DogLeg)
        optimizeDogLeg(state);

#ifdef _OUTPUT_MESSAGES_
        switch (method_)
        {
            case GaussNewton:
                cout<<"[message] With method of GaussNetow, ";               
                break;
            case LevenbergMarquardt:
                cout<<"[message] With method of LevenbergMarquardt, ";
                break;
            case DogLeg:
                cout<<"[message] With method of DogLeg, ";
                break;
            default:
                cout<<"What happened?";
                break;
        }
    if(converged_ == true)
        cout<<"it's converged! "<<endl;
    else
        cout<<"it's not converged to "<<epsilon_<<"! "<<endl;
#endif

    if(converged_ == true)
        return true;
    else
        return false;
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::optimizeGaussNetow(StateType& state)
{
    reset();
    computeResidual(state);

    while(num_iter_ < num_iter_max_ && !stop_)
    {
        startIteration();
        computeJacobian(state);
        
        StateType state_new;
        // old chi2
        chi2_ = getChi2(false);
        // calculate H
        hessian_ = jacobian_.transpose()*jacobian_;

        if( !solve())
        {
            stop_ = true;        
        }
        update(state, state_new);
        // NOTE: this will change residual
        computeResidual(state_new); 
        
        // new chi2
        double chi2_new = getChi2(false);
        num_obser_ = jacobian_.rows();

        if(chi2_ > chi2_new && !stop_)
        {
            state = state_new;
            chi2_ = chi2_new;
        }
        else
        {
            stop_ = true;
            converged_ = false;
        }
        
        // converged condition 
        // TODO: can add F(x_k+1) - F(x_K) condition
        if( maxFabs(delta_x_)< epsilon_)
        {
            converged_=true;
            stop_ = true;
        }
        ++num_iter_;
        finishIteration();
    }
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::optimizeLevenbergMarquardt(StateType& state)
{
    reset();
    computeResidual(state);

    while(num_iter_ < num_iter_max_ && !stop_)
    {
        startIteration();
        computeJacobian(state);

        StateType state_new;
        Matrix<double, Dynamic, 1> residual_old;
        double chi2_new = 0;  
        double chi2_linear = 0;

        chi2_ = getChi2(false);
        hessian_ = jacobian_.transpose()*jacobian_;
        // initialize lambda
        if(lambda_ < 0)
        {
            double tau = 1e-4;
            lambda_ = tau*maxFabs(hessian_.diagonal());
        }
        //! (H+lambda*I)
        hessian_ += lambda_*Matrix<double, D, D>::Identity();
        num_obser_ = jacobian_.rows();

        if(!solve())
        {
            rho_ = -1;  //sloved failed
        }
        else
        {
            chi2_linear = getChi2(true);    // compute linearlized error
            update(state, state_new);
            residual_old = residual_;       // save old residual to prepare for roll-back
            computeResidual(state_new);
            chi2_new = getChi2(false);      // compute original error
            // modelFidelity
            rho_ = (chi2_ - chi2_new) / (chi2_ - chi2_new);
        }
        
        if(rho_ > 0) // this step is succcess
        {   
            state = state_new;
            chi2_ = chi2_new;
            if(rho_ > min_fidelity_)
                decreaseLambda(rho_);
            else
                increaseLambda();
        }
        else  // this step is failed 
        {
            if(lambda_ > lambda_max_) // beyond upper bound and stop
            {
                stop_ = true;
            }
            else  // increase lambda and roll back, try again
            {
                residual_ = residual_old;
                increaseLambda();
            }
        }
        // whether is converged
        if(maxFabs(delta_x_) < epsilon_)
        {
            converged_ = true;
            stop_ = true;
        }

        ++num_iter_;
        finishIteration();
    }
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::increaseLambda()
{
    lambda_ *= lambda_factor_;
    lambda_factor_ *= 2;
}
template<int D, typename T>
void vo_kit::Optimizer<D, T>::decreaseLambda(double& rho)
{
    // from GTSAM3.2.1
    lambda_ *= std::max(1.0/3.0, 1.0-pow(2.*rho - 1., 3));
    lambda_factor_ = 2;
    lambda_ = std::max(lambda_, lambda_min_);
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::optimizeDogLeg(StateType& state)
{
}

template<int D, typename T> 
void vo_kit::Optimizer<D, T>::startIteration()
{
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::finishIteration()
{
#ifdef _OUTPUT_MESSAGES_ 
        switch (method_)
        {
            case GaussNewton:
                cout<<"[message]"
                    <<"\t NO."<<num_iter_
                    <<"\t Obser "<<num_obser_
                    <<"\t Chi2 "<<chi2_
                    <<"\t Stop "<<boolalpha<<stop_
                    <<"\t Converged "<<boolalpha<<converged_
                    <<endl;               
                break;
            case LevenbergMarquardt:
                cout<<"[message]"
                    <<"\t NO."<<num_iter_
                    <<"\t Obser "<<num_obser_
                    <<"\t Chi2 "<<chi2_
                    <<"\t rho "<<rho_
                    <<"\t Lambda "<<lambda_
                    <<"\t Factor "<<lambda_factor_
                    <<"\t Stop "<<boolalpha<<stop_
                    <<"\t Converged "<<boolalpha<<converged_
                    <<endl;               
                break;
            case DogLeg:
                cout<<"[message]";
                break;
            default:
                cout<<"What happened?";
                break;
        }
#endif    
}


// #endif