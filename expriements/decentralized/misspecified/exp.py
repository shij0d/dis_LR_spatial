#%%
#import sys


## Add the path where your Python packages are located
#sys.path.append('/home/shij0d/documents/dis_LR_spatial')

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

#%%

def estimate(r):
    alpha=1
    length_scale=0.1
    nu=0.5
    N=10000
    mis_dis=0.02
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=2
    J=10
    con_pro=0.5
    er = generate_connected_erdos_renyi_graph(J, con_pro)
    adj_matrix=nx.adjacency_matrix(er).todense()
    np.fill_diagonal(adj_matrix, 1)
    weights,_=optimal_weight_matrix(adj_matrix=adj_matrix)
    weights=torch.tensor(weights,dtype=torch.double)
    #weights = torch.ones((J,J),dtype=torch.float64)/J

    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
    data,knots=sampler.generate_obs_gp(m=100,method="random")
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
    try:
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        print("global optimization succeed")
    except Exception:
        optimal_estimator=(r, "global minimization error")
        print("global optimization failed")
    
    try:
        mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_true)
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
    mu_list=[]
    Sigma_list=[]
    beta_list=[]
    delta_list=[]
    theta_list=[]
    for j in range(J):
        mu_list.append(mu)
        Sigma_list.append(Sigma)
        beta_list.append(beta)
        delta_list.append(delta)
        theta_list.append(theta)


    T=100
    try:
        de_estimators=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T,weights_round=6)
        print("dis optimization succeed")
    except Exception:
        print("dis optimization failed")
        return (r, "distributed minimization error")
    return de_estimators,optimal_estimator


rs=[i for i in range(100)]
results = Parallel(n_jobs=-1)(delayed(estimate)(r) for r in rs)
with open(f'expriements/decentralized/misspecified/results_memeff.pkl', 'wb') as f:
    pickle.dump(results, f)