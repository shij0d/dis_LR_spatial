
# what is the next step
# 1. determine the area to be analyzed: done (north american)
# 2. Partition the data into two parts: done
# 3. One part is for parameter estimation, then predict the value in the other part
# # 3.1 Which distance function should I use: just use the chordal distance induced by the Euclidean distance (refer literature)
# # 3.2 incorporate the intercept: yes
# 4. compare the decentralized method with MLE

# 5. compare with the method in the paper in terms of computational efficiency, estimation accuracy and prediction accuracy?

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
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel,onedif_kernel,matern_kernel_factory
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


path = "/home/shij0d/Documents/Dis_Spatial/real_data/Blended-Hydro_TPW_MAP_d20241105/"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
file_path=path+files[0]
file2read = Dataset(file_path)


lats=file2read.variables['lat'][:]
lons=file2read.variables['lon'][:]
TPWs=file2read.variables['TPW'][:]
observation_ages=file2read.variables['observation_age'][:]
satellite_numbers=file2read.variables['Satellite_Number'][:]
quality_informations=file2read.variables['quality_information'][:]
quality_flags=file2read.variables['quality_flag'][:]
lon_grid, lat_grid = np.meshgrid(lons.data, lats.data)



#select the data 
lat_min, lat_max = 15, 75  # Latitude range for North America
lon_min, lon_max = -170, -50  # Longitude range for North America


lat_min, lat_max = 15, 75  # Latitude range for North America
lon_min, lon_max = -170, -50  # Longitude range for North America

lon_grid, lat_grid = np.meshgrid(lons.data, lats.data)

region_mask = (
        (lat_grid >= lat_min) & (lat_grid <= lat_max) &
        (lon_grid >= lon_min) & (lon_grid <= lon_max)
    )
lats_NorthAmerican = lat_grid[region_mask]
lons_NorthAmerican = lon_grid[region_mask]
TPWs_NorthAmerican = TPWs[region_mask]
satellite_numbers_NorthAmerican =satellite_numbers[region_mask]

unique_satellites, counts = np.unique(satellite_numbers_NorthAmerican, return_counts=True)

# Display results
for satellite, count in zip(unique_satellites, counts):
    print(f"Satellite Number {satellite}: {count} observations")
unique_satellites=unique_satellites.compressed()
dis_data=[]
#divide the data according to sat
for sat in unique_satellites:
    mask = satellite_numbers_NorthAmerican == sat
    if sat==103.0:
        continue
    lats_sat = lats_NorthAmerican[mask]
    lons_sat = lons_NorthAmerican[mask]
    TPWs_sat = TPWs_NorthAmerican[mask]
    valid_indices = ~TPWs_sat.mask
    # Filter the arrays
    lats_sat = lats_sat[valid_indices]
    lons_sat = lons_sat[valid_indices]
    TPWs_sat = TPWs_sat[valid_indices]
    X=np.ones((lats_sat.shape[0],1))
    local_data=np.hstack((lats_sat.reshape(-1,1),lons_sat.reshape(-1,1),TPWs_sat.reshape(-1,1),X))
    dis_data.append(local_data)


#estimation
def estimation(nu,min_dis):
    J=len(dis_data)
    con_pro=0.5
    er = generate_connected_erdos_renyi_graph(J, con_pro)
    adj_matrix=nx.adjacency_matrix(er).todense()
    np.fill_diagonal(adj_matrix, 1)
    weights,_=optimal_weight_matrix(adj_matrix=adj_matrix)
    weights=torch.tensor(weights,dtype=torch.double)
    lats=np.arange(lat_min, lat_max,min_dis)
    lons=np.arange(lon_min, lon_max,min_dis)
    knots = [(lat, lon) for lat in lats for lon in lons]
    #weights = torch.ones((J,J),dtype=torch.float64)/J

    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, onedif_kernel, knots, weights)
    else:
        matern_kernel_nu=matern_kernel_factory(nu)
        gpp_estimation = GPPEstimation(dis_data, matern_kernel_nu, knots, weights)
    
    beta_ini=torch.tensor(1,dtype=torch.float64)
    delta_ini=torch.tensor(1,dtype=torch.float64)
    alpha_ini=1
    length_scale_ini=0.5
    theta_ini=torch.tensor([alpha_ini,length_scale_ini],dtype=torch.float64)
    x_ini=gpp_estimation.argument2vector_lik(beta_ini,delta_ini,theta_ini)
    
    
    
    
    try:
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_ini)
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        print("global optimization succeed")
        print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    except Exception:
        optimal_estimator=("global minimization error")
        print("global optimization failed")
  
    # try:

    #     mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_ini,job_num=-1)

    #     print("local optimization succeed")
    # except Exception:
    #     print("local optimization failed")
    #     return ("local minimization error")
    # if len(mu_list)==0:
    #     print("local optimization failed")
    #     return ("local minimization error",optimal_estimator)   
    
    # mu=mu_list[0]
    # Sigma=Sigma_list[0]
    # beta=beta_list[0]
    # delta=delta_list[0]
    # theta=theta_list[0]
    # num=len(mu_list)
    # if num>1:
    #     for j in range(1,num):
    #         mu+=mu_list[j]
    #         Sigma+=Sigma_list[j]
    #         beta+=beta_list[j]
    #         delta+=delta_list[j]
    #         theta+=theta_list[j]
    # mu=mu/num
    # Sigma=Sigma/num
    # beta=beta/num
    # delta=delta/num
    # theta=theta/num
    # print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
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
    #de_estimators=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T,weights_round=6)
    # try:
    #     de_estimators=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T,weights_round=6)
    #     print("dis optimization succeed")
    # except Exception:
    #     print("dis optimization failed")
    #     return ("distributed minimization error",optimal_estimator)
  
    #return de_estimators,optimal_estimator

result=estimation(0.5,10)

file_path_save="/home/shij0d/Documents/Dis_Spatial/real_data/result.pkl"
with open(file_path_save, "wb") as file:
    pickle.dump(result, file)