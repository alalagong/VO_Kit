#ifndef _OPTIMIZER_BASE_HPP_
#define _OPTIMIZER_BASE_HPP_

template<int D, typename T>
bool vo_kit::Optimizer<D, T>::runOptimize(StateType& state)
{  
#ifdef _OUTPUT_MESSAGES_
        switch (method_)
        {
            case GaussNewton:
                cout<<"[message] With method of GaussNetow ";               
                break;
            case LevenbergMarquardt:
                cout<<"[message] With method of LevenbergMarquardt ";
                break;
            case DogLeg:
                cout<<"[message] With method of DogLeg ";
                break;
            default:
                cout<<"What happened?";
                break;
        }
        cout<<endl;
#endif

    if(method_ == GaussNewton)
        optimizeGaussNetow(state);
    else if(method_ == LevenbergMarquardt)
        optimizeLevenbergMarquardt(state);
    else if(method_ == DogLeg)
        optimizeDogLeg(state);
        
    if(converged_ == true)
        return true;
    else
        return false;
}

//* this will also be used in Dog-Leg
template<int D, typename T>
void vo_kit::Optimizer<D, T>::optimizeGaussNetow(StateType& state)
{
    if(method_ != DogLeg)
    {
        reset();
        computeResidual(state);
    }
    while(num_iter_ < num_iter_max_ && !stop_)
    {
        if(method_ != DogLeg)
            startIteration();
        computeJacobian(state);
        
        StateType state_new(state);
        // new chi2
        double chi2_new = 0;
        // old chi2
        chi2_ = getChi2(false);
        // calculate H
        hessian_ = jacobian_.transpose()*jacobian_;
        jres_ = jacobian_.transpose()*residual_;

        if( !solve())
        {
            stop_ = true;
            if(method_ == DogLeg)
                return;
        }
        else
        {
            if(method_ == DogLeg)
                return;

            update(state, state_new);
            // NOTICE: this will change residual
            computeResidual(state_new); 
            chi2_new = getChi2(false);
        }
        
        num_obser_ = jacobian_.rows();

        if(chi2_ >= chi2_new && !stop_)
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
        if( utils::maxFabs(delta_x_) < epsilon_ && !std::isnan(delta_x_[0]))
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
        jres_ = jacobian_.transpose()*residual_;

        // initialize lambda
        if(lambda_ < 0)
        {
            double tau = 1e-4;
            lambda_ = tau*utils::maxFabs(hessian_.diagonal());
        }
        //! (H+lambda*I)
        hessian_ += lambda_*Matrix<double, D, D>::Identity();

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
// #ifdef _DEBUG_MODE_
//             std::cout<<"[Debug] "
//                 <<"\t chi2"<<chi2_
//                 <<"\t chi2_new "<<chi2_new
//                 <<"\t chi2_linear"<<chi2_linear
//                 <<std::endl;
// #endif
            // modelFidelity
            rho_ =(chi2_ - chi2_new)/(chi2_ - chi2_linear);
        }

        num_obser_ = jacobian_.rows();
        
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
        if(utils::maxFabs(delta_x_) < epsilon_)
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
    reset();
    computeResidual(state);

    while(num_iter_ < num_iter_max_ && !stop_)
    {
        startIteration();

        StateType state_new(state);
        double chi2_new = 0, chi2_linear = 0;
        Matrix<double, Dynamic, 1> residual_old;
        
        computeDeltaGN(state);  // delta_GN
        computeDeltaSD();       // delta_SD
        computeDeltaDL();       // delta_x_

        chi2_linear = getChi2(true);
        residual_old = residual_;
        update(state, state_new);
        computeResidual(state_new);
        chi2_new = getChi2(false);
        // NOTE: for Dog-Leg, the method of getting chi2_linear maybe not good
        rho_ = fabs(chi2_ - chi2_new) < 1e-15 || fabs(chi2_ - chi2_linear) < 1e-15 ?
                0.5 : (chi2_ - chi2_new)/(chi2_ - chi2_linear);
        num_obser_ = residual_.rows();

        if(rho_ > 0.75)
        {
            double delta_x_norm = delta_x_.norm();
            state = state_new;
            Delta_ = std::max(Delta_, 3.f*delta_x_norm);
        }
        else if( rho_ < 0.25)
        {
            Delta_ /= 2;
            Delta_ = std::max(Delta_, 1e-5);

            if(rho_ > 0)
                state = state_new;
            else
                residual_ = residual_old; // keep old state and roll back
            // can't continue
            if(rho_ < 0 && Delta_ == 1e-5)
                stop_ = true;
        }
        else
        {
            assert(rho_ <= 0.75 && rho_ >= 0.25);
            state = state_new;
        }
        

        if(utils::maxFabs(delta_x_) < epsilon_)
        {
            stop_ = true;
            converged_ = true;
        }

        ++num_iter_;
        finishIteration();
    }

}

template<int D, typename T> 
void vo_kit::Optimizer<D, T>::startIteration()
{
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::finishIteration()
{
#ifdef _DEBUG_MODE_
    // cout<<"[Debug] "
        // <<"\t H "<<hessian_
        // <<"\t residual "<< residual_
        // <<"\t jacobian "<<jacobian_
        // <<"\t delta_x "<< delta_x_.transpose()
        // <<endl;
#endif
#ifdef _OUTPUT_MESSAGES_ 
        switch (method_)
        {
            case GaussNewton:
                std::cout<<"[message]"<<std::setprecision(4)
                    <<"\t NO."<<num_iter_
                    <<"\t Obser "<<num_obser_
                    <<"\t Chi2 "<<chi2_
                    <<"\t delta_x "<<delta_x_.transpose()
                    <<"\t Stop "<<boolalpha<<stop_
                    <<"\t Converged "<<boolalpha<<converged_
                    <<std::endl;               
                break;
            case LevenbergMarquardt:
                std::cout<<"[message]"<<std::setprecision(4)
                    <<"\t NO."<<num_iter_
                    <<"\t Obser "<<num_obser_
                    <<"\t Chi2 "<<chi2_
                    <<"\t delta_x "<<delta_x_.transpose()
                    <<"\t rho "<<rho_
                    <<"\t Lambda "<<lambda_
                    <<"\t Factor "<<lambda_factor_
                    <<"\t Stop "<<boolalpha<<stop_
                    <<"\t Converged "<<boolalpha<<converged_
                    <<std::endl;               
                break;
            case DogLeg:
                std::cout<<"[message]"<<std::setprecision(4)
                    <<"\t NO."<<num_iter_
                    <<"\t Obser "<<num_obser_
                    <<"\t Chi2 "<<chi2_
                    <<"\t delta_x "<<delta_x_.transpose()
                    <<"\t rho "<<rho_
                    <<"\t beta "<<beta_
                    <<"\t Delta "<<Delta_
                    <<"\t Stop "<<boolalpha<<stop_
                    <<"\t Converged "<<boolalpha<<converged_
                    <<std::endl;
                break;
            default:
                std::cout<<"What happened?";
                break;
        }
#endif    
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::computeDeltaGN(StateType& state)
{
    // this will compute jacobian_, hessian_, jres_, chi2_
    optimizeGaussNetow(state);
    delta_GN_x_ = delta_x_;
}

template<int D, typename T>
void vo_kit::Optimizer<D, T>::computeDeltaSD()
{
    const double jres_norm_sq = jres_.squaredNorm();
    const double jresT_H_jres = jres_.transpose()*hessian_*jres_;
    const double step_length = jres_norm_sq / jresT_H_jres;

    delta_SD_x_ = - step_length*jres_;
}

template<int D, typename T> 
void vo_kit::Optimizer<D, T>::computeDeltaDL()
{
    assert(Delta_ >= 0.0);
    // GN failed
    if(stop_ != true)
    {
        const double Delta_square = Delta_*Delta_;
        const double delta_GN_norm_sq = delta_GN_x_.squaredNorm();
        const double delta_SD_norm_sq = delta_SD_x_.squaredNorm();

        if(delta_GN_norm_sq <= Delta_square) // choose GN
        {
            delta_x_ = delta_GN_x_;
#ifdef _DEBUG_MODE_
            std::cout<<"use GN delta_x"<<std::endl;
#endif
        }
        else if(delta_SD_norm_sq >= Delta_square) // choose SD direction
        {
            delta_x_ =  std::sqrt(Delta_square / delta_SD_norm_sq) * delta_SD_x_;
#ifdef _DEBUG_MODE_
            std::cout<<"use SD delta_x"<<std::endl;
#endif
        }   
        else
        {
            assert(delta_GN_norm_sq > Delta_square);
            assert(delta_SD_norm_sq < Delta_square);
            computeBeta();
            delta_x_ = (1-beta_)*delta_SD_x_ + delta_GN_x_;
        }
        stop_ = false;
    }
    else // only use SD
    {
#ifdef _DEBUG_MODE_
            std::cout<<"can't get GN deltax, so use SD delta_x"<<std::endl;
#endif
        delta_x_ = delta_SD_x_;
        stop_ = false;
    }
    
}

// from GTSAM3.2.1
template<int D, typename T>
void vo_kit::Optimizer<D, T>::computeBeta()
{
    // inner products
    const double v_GN_sq = delta_GN_x_.dot(delta_GN_x_);
    const double v_SD_sq = delta_SD_x_.dot(delta_SD_x_);
    const double v_GN_SD = delta_GN_x_.dot(delta_SD_x_);
    const double Delta_square = Delta_*Delta_;

    // the coefficient of quadratic formula (ax^2+bx+c)
    const double a = v_GN_sq + v_GN_sq - 2*v_GN_SD;
    const double b = 2*(v_GN_SD - v_SD_sq);
    const double c = v_SD_sq - Delta_square;
    const double b2_4ac_sqrt = std::sqrt(b*b - 4*a*c);

    // two possible value of beta
    double beta1, beta2;
    beta1 = (-b + b2_4ac_sqrt) / (2*a);
    beta2 = (-b - b2_4ac_sqrt) / (2*a);

    // right beta
    if(beta1 >= 0.f && beta1 <= 1.f)
    {
        assert(!(0.0 <= beta2 && beta2 <= 1));
        beta_ = beta1;
    } else
    {
        assert((0.0 <= beta2 && beta2 <= 1));
        beta_ = beta2;
    }
    
}

#endif // end _OPTIMIZER_BASE_HPP_