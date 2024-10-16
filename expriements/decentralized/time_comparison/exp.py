#%%
import sys


# Add the path where your Python packages are located
sys.path.append('/home/shij0d/documents/dis_LR_spatial')

import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel,onedif_kernel
from src.networks import generate_connected_erdos_renyi_graph
from src.weights import optimal_weight_matrix
import networkx as nx
import numpy as np
import random
import pickle
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
#%%



def estimate(r,J):
    alpha=1
    length_scale=0.1
    nu=0.5
    n=10000
    N=n*J
    mis_dis=0.02
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=2
    weights = torch.ones((J,J),dtype=torch.float64)/J

    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
    data,knots=sampler.generate_obs_gpp(m=100,method="random")
    dis_data=sampler.data_split(data,J)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, onedif_kernel, knots, weights)
    else:
        raise("incompleted")
    
    beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
    delta=torch.tensor(0.25,dtype=torch.float64)
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    x_true=gpp_estimation.argument2vector_lik(beta,delta,theta)
    
    start_time = time.time()
    mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true,thread_num=1)
    optimal_estimator=(mu,Sigma,beta,delta,theta,result)
    end_time = time.time()
    elapsed_time_mle = end_time - start_time
    print(theta)
    print(elapsed_time_mle)
    
    start_time = time.time()
    mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_true,J,thread_num=1)
    
    mu=mu_list[0]
    Sigma=Sigma_list[0]
    beta=beta_list[0]
    delta=delta_list[0]
    theta=theta_list[0]
    num=len(mu_list)
    if num>1:
        for j in range(1,num):
            mu+=mu_list[j]
            Sigma+=Sigma_list[j]
            beta+=beta_list[j]
            delta+=delta_list[j]
            theta+=theta_list[j]
    mu=mu/num
    Sigma=Sigma/num
    beta=beta/num
    delta=delta/num
    theta=theta/num
   

    T=32

    de_estimators=gpp_estimation.ce_optimize_stage2(mu,Sigma,beta,delta,theta,T,J,thread_num=1)
    end_time = time.time()
    elapsed_time_de = end_time - start_time
    print(elapsed_time_de)
    return elapsed_time_mle,elapsed_time_de,de_estimators,optimal_estimator

Js=[10,20,40]
results=[]
for J in Js:
    results_r=[]
    for r in range(10):
        result=estimate(r,J)
        results_r.append(result)
    results.append(results_r)
    
