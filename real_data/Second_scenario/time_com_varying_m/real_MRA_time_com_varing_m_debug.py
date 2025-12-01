#%%
path_project='/home/shij0d/Documents/Dis_Spatial'
import os
import sys
import time

# Add the path where your Python packages are located
sys.path.append(path_project)

import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch_real_data import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel,onedif_kernel,matern_kernel_factory,matern_kernel
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
import pandas as pd

SEED=2024
path=os.path.join(path_project,'real_data/Second_scenario/MRA_codeAndData/MIRSmra.csv')
print(path)
full_data=pd.read_csv(path,header=None,index_col=None).values

full_data[:,0]=full_data[:,0]-180
full_data[:, [0, 1]] = full_data[:, [1, 0]]

#full_data=full_data[:200000,]



beta=np.mean(full_data[:,2])
full_data[:, 2] -= beta
full_data=full_data[:,:3]
locations=full_data[:,:2]
locations = [tuple(row) for row in locations]

np.random.seed(42)  # Set seed for reproducibility
shuffled_data = np.random.permutation(full_data)  # Shuffle rows of the array
# num_parts = 20
# dis_data = np.array_split(shuffled_data, num_parts)


#initialized with small sample size
def initial(nu,knots):
    
    N=500
    J=1
    nu=1.5
    data=shuffled_data[:N,:]
    weights = torch.ones((J,J),dtype=torch.float64)/J
    dis_data = np.array_split(data, J)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        #matern_kernel_nu=matern_kernel_factory(nu)
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)
        
        
    beta_ini=None
    delta_ini=torch.tensor(1,dtype=torch.float64)
    alpha_ini=10
    length_scale_ini=0.5
    theta_ini=torch.tensor([alpha_ini,length_scale_ini],dtype=torch.float64)
    x_ini=gpp_estimation.argument2vector_lik(beta_ini,delta_ini,theta_ini)
    mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_ini)
    inital_estimator=(mu,Sigma,beta,delta,theta,result) 
    return inital_estimator





#estimation
def estimation(r,m=60,if_optimal=True,J=16,nu=1.5):
    dis_data = np.array_split(shuffled_data, J)
    J=len(dis_data)
    weights = torch.ones((J,J),dtype=torch.float64)/J
    # #grid knots
    # min_dis=3
    # lat_min,lat_max=torch.min(locations[:,0]),torch.max(locations[:,0])
    # lon_min,lon_max=torch.min(locations[:,0]),torch.max(locations[:,0])
    # lats=np.arange(lat_min, lat_max,min_dis)
    # lons=np.arange(lon_min, lon_max,min_dis)
    # knots = [(lat, lon) for lat in lats for lon in lons]
    #random knots
    random.seed(SEED+r)
    knots = random.sample(locations, m)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)  
    
    random.seed(SEED+1)
    knots_inital = random.sample(locations, 60)
    #knots_inital=knots
    initial_estimator=initial(nu,knots_inital)
    x_inital=gpp_estimation.argument2vector_lik(initial_estimator[2],initial_estimator[3],initial_estimator[4])
    if if_optimal:
        start_time = time.time()
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_inital)
        end_time = time.time()
        elapsed_time_mle = end_time - start_time
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        #print(f"delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
        #print(result)
        print(elapsed_time_mle)
    else:
        optimal_estimator=[None for i in range(6)]
        elapsed_time_mle=None

    return optimal_estimator
# m=60
# optimal_estimator=estimation(m=m)
# print(f"m={m}")
# print(optimal_estimator[-1])

# %%
ms=[60,70,80,90,100]
r=0
for m in ms:
    optimal_estimator=estimation(r,m=m)
    print(f"m={m}")
    print(optimal_estimator[-1])
# %%
