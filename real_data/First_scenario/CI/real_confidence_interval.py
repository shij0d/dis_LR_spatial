
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
import pandas as pd
import scipy.stats as stats


import sys
import time

# Add the path where your Python packages are located
#sys.path.append('/home/shij0d/Documents/Dis_Spatial')

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

SEED=2024



path = "real_data/Blended-Hydro_TPW_MAP_d20241105/"
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


lat_min, lat_max = 15, 55  
lon_min, lon_max = -160, -120  

#United states

lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.0,-66.93457
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
    # if sat==501.0:
    #     continue
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
    
    #withoutX
    local_data=np.hstack((lats_sat.reshape(-1,1),lons_sat.reshape(-1,1),TPWs_sat.reshape(-1,1)))
    # np.random.seed(SEED)
    # local_shuffled_data = np.random.permutation(local_data)
    # local_data=local_shuffled_data[:20,:]
    dis_data.append(local_data)
    
#randomly 
full_data=np.vstack(dis_data)#.data
size=full_data.shape[0]
#full_data=full_data[:12000,:]
#remove the mean
beta=np.mean(full_data[:,2])
full_data[:, 2] -= beta
locations=full_data[:,:2]
locations = [tuple(row) for row in locations]

for i,local_data in enumerate(dis_data):
    local_data[:,2]-=beta
    dis_data[i]=local_data


# np.random.seed(42)  # Set seed for reproducibility
# shuffled_data = np.random.permutation(full_data)  # Shuffle rows of the array
# num_parts = 5
# dis_data = np.array_split(shuffled_data, num_parts)


#estimation
def construct(nu):
    J=len(dis_data)
    con_pro=0.5
    er = generate_connected_erdos_renyi_graph(J, con_pro)
    adj_matrix=nx.adjacency_matrix(er).todense()
    np.fill_diagonal(adj_matrix, 1)
    weights,_=optimal_weight_matrix(adj_matrix=adj_matrix)
    weights=torch.tensor(weights,dtype=torch.double)
    random.seed(SEED)
    m=100
    knots = random.sample(locations, m)
    
    if nu==0.5:
        gpp_estimation = GPPEstimation(dis_data, partial(exponential_kernel,type="chordal"), knots, weights)
    elif nu==1.5:
        gpp_estimation = GPPEstimation(dis_data, partial(onedif_kernel,type="chordal"), knots, weights)
    else:
        gpp_estimation = GPPEstimation(dis_data, partial(matern_kernel,nu=nu,type="chordal"), knots, weights)
    
    return gpp_estimation

gpp_estimation=construct(1.5)

alpha = 0.05  # Significance level (e.g., 95% confidence level)
z_alpha_half = stats.norm.ppf(1 - alpha / 2)  # Critical value

file_path="real_data/result.pkl"
with open(file_path, "rb") as file:
    de_estimators, optimal_estimator = pickle.load(file)

#non-distributed
mu,Sigma,beta,delta,theta,_=optimal_estimator
asy_variance_optimal=gpp_estimation.ce_asy_variance_autodif(beta,delta,theta,job_num=-1)
m=gpp_estimation.m
V_delta=asy_variance_optimal[0]
V_theta=asy_variance_optimal[1]
Inv_V_theta=torch.inverse(V_theta)
CI_delta=(delta-z_alpha_half/math.sqrt((V_delta*size)),delta+z_alpha_half/math.sqrt((V_delta*size)))
CI_theta_0=(theta[0]-z_alpha_half*math.sqrt(Inv_V_theta[0,0]/m),theta[0]+z_alpha_half*math.sqrt(Inv_V_theta[0,0]/m))
CI_theta_1=(theta[1]-z_alpha_half*math.sqrt(Inv_V_theta[1,1]/m),theta[1]+z_alpha_half*math.sqrt(Inv_V_theta[1,1]/m))

#distributed
mu_list,Sigma_list,beta_lists,delta_lists,theta_lists,s_list,f_value=de_estimators
beta_list,delta_list,theta_list=beta_lists[-1],delta_lists[-1],theta_lists[-1]
CI_delta_list=[]
CI_theta_0_list=[]
CI_theta_1_list=[]
for j in range(gpp_estimation.J):
    beta_j=beta_list[j]
    delta_j=delta_list[j]
    theta_j=theta_list[j]
    asy_variance_j=gpp_estimation.ce_asy_variance_autodif(beta_j,delta_j,theta_j,job_num=-1)
    V_delta_j=asy_variance_j[0]
    V_theta_j=asy_variance_j[1]
    Inv_V_theta_j=torch.inverse(V_theta)
    CI_delta_j=(delta_j-z_alpha_half/math.sqrt((V_delta_j*size)),delta_j+z_alpha_half/math.sqrt((V_delta_j*size)))
    CI_theta_0_j=(theta_j[0]-z_alpha_half*math.sqrt(Inv_V_theta_j[0,0]/m),theta_j[0]+z_alpha_half*math.sqrt(Inv_V_theta_j[0,0]/m))
    CI_theta_1_j=(theta_j[1]-z_alpha_half*math.sqrt(Inv_V_theta_j[1,1]/m),theta_j[1]+z_alpha_half*math.sqrt(Inv_V_theta_j[1,1]/m))
    CI_delta_list.append(CI_delta_j)
    CI_theta_0_list.append(CI_theta_0_j)
    CI_theta_1_list.append(CI_theta_1_j)


#tranform to the corresponding intervals to the paper

#columns: tau_lower, tau_upper, sigma_lower, sigma_upper, beta_lower, beta_upper
#rows: non_distributed, machine 1,.., machine 6


# Define columns and rows
columns = ["tau_lower", "tau_upper", "sigma_lower", "sigma_upper", "beta_lower", "beta_upper"]
rows = ["non_distributed"] + [f"machine {i+1}" for i in range(gpp_estimation.J)]

# Create the DataFrame
CI_df = pd.DataFrame(index=rows, columns=columns)

# Fill in values for non-distributed method
CI_df.loc["non_distributed", "tau_lower"] = 1 / math.sqrt(CI_delta[1])
CI_df.loc["non_distributed", "tau_upper"] = 1 / math.sqrt(CI_delta[0])
CI_df.loc["non_distributed", "sigma_lower"] = math.sqrt(CI_theta_0[0])
CI_df.loc["non_distributed", "sigma_upper"] = math.sqrt(CI_theta_0[1])
CI_df.loc["non_distributed", "beta_lower"] = CI_theta_1[0].item()
CI_df.loc["non_distributed", "beta_upper"] = CI_theta_1[1].item()

# Fill in values for distributed method
for j in range(gpp_estimation.J):
    CI_df.loc[f"machine {j+1}", "tau_lower"] = 1 / math.sqrt(CI_delta_list[j][1])
    CI_df.loc[f"machine {j+1}", "tau_upper"] = 1 / math.sqrt(CI_delta_list[j][0])
    CI_df.loc[f"machine {j+1}", "sigma_lower"] = math.sqrt(CI_theta_0_list[j][0])
    CI_df.loc[f"machine {j+1}", "sigma_upper"] = math.sqrt(CI_theta_0_list[j][1])
    CI_df.loc[f"machine {j+1}", "beta_lower"] = CI_theta_1_list[j][0].item()
    CI_df.loc[f"machine {j+1}", "beta_upper"] = CI_theta_1_list[j][1].item()

# Save the DataFrame to a CSV file
output_path = "real_data/CI.csv"
CI_df.to_csv(output_path)

print(f"Confidence interval data saved to {output_path}")

    


