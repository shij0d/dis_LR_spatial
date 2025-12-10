############Note: This experiment is done over the machine parrot############

#%%
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
import os
#%%

def initial():
    alpha=1
    length_scale=0.1
    nu=0.5
    N=2000
    J=1
    mis_dis=0.02
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=2
    weights = torch.ones((J,J),dtype=torch.float64)/J

    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=3232134)
    data,knots=sampler.generate_obs_gpp(m=50,method="random")
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
    mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
    inital_estimator=(mu,Sigma,beta,delta,theta,result) 
    return inital_estimator


initial_estimator=initial()

def estimate(r,n):
    alpha=1
    length_scale=0.1
    nu=0.5
    #n=10000
    J=16
    N=n*J
    mis_dis=0.02
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=2
    weights = torch.ones((J,J),dtype=torch.float64)/J

    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=2024)
    data,knots=sampler.generate_obs_gpp(m=100,method="random")
    dis_data=sampler.data_split(data,J)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, onedif_kernel, knots, weights)
    else:
        raise("incompleted")
    
   
    x_inital=gpp_estimation.argument2vector_lik(initial_estimator[2],initial_estimator[3],initial_estimator[4])
    start_time = time.time()
    mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_inital,thread_num=1)
    optimal_estimator=(mu,Sigma,beta,delta,theta,result)
    end_time = time.time()
    elapsed_time_mle = end_time - start_time
    print(theta)
    print(elapsed_time_mle)
        
   

    T=32
    start_time = time.time()
    de_estimators=gpp_estimation.ce_optimize_stage2(mu,Sigma,beta,delta,theta,T,J,thread_num=1)
    end_time = time.time()
    elapsed_time_de = end_time - start_time
    print(elapsed_time_de)
    return elapsed_time_mle,elapsed_time_de,de_estimators,optimal_estimator

#Limit the number of threads for numpy, OpenBLAS, or MKL (if used)
os.environ["OMP_NUM_THREADS"] = "1"  # For OpenMP (e.g., NumPy, scikit-learn)
os.environ["MKL_NUM_THREADS"] = "1"  # If you're using MKL-based libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS (NumPy, SciPy)
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr (if used)
ns=[10**3,5*10**3,10**4,5*10**4]
results=[]
for n in ns:
    results_r=[]
    for r in range(20):
        print(f"N:{16*n},r:{r}")
        result=estimate(r,n)
        results_r.append(result)
    results.append(results_r)
    with open(f'expriements/decentralized/time_comparison/varying_N_fixed_J.pkl', 'wb') as f:
        pickle.dump(results, f)