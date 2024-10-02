import sys


# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')

import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel
import numpy as np
import random
import pickle
from functools import partial
import multiprocessing



def estimate(r,length_scale):
    alpha=1
    #length_scales=[0.3,0.1,0.03]
    nu=0.5
    N=10000
    mis_dis=0.001
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=2
    J=10
    
    
    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
    data,knots=sampler.generate_obs_gpp(m=100,method="random")
    dis_data=sampler.data_split(data,J)
    weights = torch.ones((J,J),dtype=torch.float64)/J
    gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
    delta=torch.tensor(0.25,dtype=torch.float64)
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    x_true=gpp_estimation.argument2vector_lik(beta,delta,theta)
    # try:
    #     mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
    #     optimal_estimator=(mu,Sigma,beta,delta,theta,result)
    #     print("global optimization succeed")
    # except Exception:
    #     optimal_estimator=(r, "global minimization error")
    #     print("global optimization failed")
    
    try:
        mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers(x_true)
        print("local optimization succeed")
    except Exception:
        print("local optimization failed")
        return (r, "local minimization error")
    if len(mu_list)==0:
        print("local optimization failed")
        return (r, "local minimization error")   
        
    mu=mu_list[0]
    Sigma=Sigma_list[0]
    beta=beta_list[0]
    delta=delta_list[0]
    theta=theta_list[0]
    
    for j in range(1,J):
        mu+=mu_list[j]
        Sigma+=Sigma_list[j]
        beta+=beta_list[j]
        delta+=delta_list[j]
        theta+=theta_list[j]
    mu=mu/J
    Sigma=Sigma/J
    beta=beta/J
    delta=delta/J
    theta=theta/J

    T=50
    weights_full = torch.ones((J,J),dtype=torch.float64)/J #
    gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights_full)

    try:
        ce_estimators=gpp_estimation.ce_optimize_stage2(mu,Sigma,beta,delta,theta,T=50)
        print("dis optimization succeed")
    except Exception:
        print("dis optimization failed")
        return (r, "distributed minimization error")
  
    return ce_estimators,optimal_estimator

estimate(7,0.3)
