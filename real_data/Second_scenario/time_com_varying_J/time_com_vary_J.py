
# what is the next step
# 1. determine the area to be analyzed: done (north american)
# 2. Partition the data into two parts: done
# 3. One part is for parameter estimation, then predict the value in the other part
# # 3.1 Which distance function should I use: just use the chordal distance induced by the Euclidean distance (refer literature)
# # 3.2 incorporate the intercept: yes
# 4. compare the decentralized method with MLE

# 5. compare with the method in the paper in terms of computational efficiency, estimation accuracy and prediction accuracy?
import warnings

# Suppress FutureWarning from torch.load
#warnings.filterwarnings("ignore")

warnings.simplefilter("error", FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np



import sys
import time

# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')

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
path="/home/shij0d/Documents/Dis_Spatial/real_data/MRA_codeAndData/MIRSmra.csv"
full_data=pd.read_csv(path,header=None,index_col=None).values

full_data[:,0]=full_data[:,0]-180
full_data[:, [0, 1]] = full_data[:, [1, 0]]

full_data=full_data[:200000,]



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
    
    N=2000
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
def estimation(r,J,nu=1.5):
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
    random.seed(SEED+2)
    m=50
    knots = random.sample(locations, m)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)  
    
    initial_estimator=initial(nu,knots)
    x_inital=gpp_estimation.argument2vector_lik(initial_estimator[2],initial_estimator[3],initial_estimator[4])
    
    start_time = time.time()
    mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_inital,thread_num=1)
    end_time = time.time()
    elapsed_time_mle = end_time - start_time
    optimal_estimator=(mu,Sigma,beta,delta,theta,result)
    
    print(f"delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    print(result)
    print(elapsed_time_mle)

    T=32
    start_time = time.time()
    de_estimators=gpp_estimation.ce_optimize_stage2(initial_estimator[0],initial_estimator[1],initial_estimator[2],initial_estimator[3],initial_estimator[4],T,J,thread_num=J,backend='threading')   
    end_time = time.time()
    elapsed_time_de = end_time - start_time
    print(elapsed_time_de)
    #return elapsed_time_mle,elapsed_time_de,de_estimators,optimal_estimator



Js=[2,4,8,10,16]

Js=[20]
os.environ["OMP_NUM_THREADS"] = "1"  # For OpenMP (e.g., NumPy, scikit-learn)
os.environ["MKL_NUM_THREADS"] = "1"  # If you're using MKL-based libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS (NumPy, SciPy)
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr (if used)
results=[]
for J in Js:
    results_r=[]
    for r in range(20):
        print(f"J:{J},r:{r}")
        result=estimation(r,J)
        results_r.append(result)
    results.append(results_r)
with open(f'/home/shij0d/Documents/Dis_Spatial/real_data/time_comparison/time_com_varying_J_fixed_N.pkl', 'wb') as f:
    pickle.dump(results, f)
    
