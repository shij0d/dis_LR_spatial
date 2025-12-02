# %%
#import sys
#sys.path.append('/home/shij0d/Documents/Dis_Spatial')

from src.kernel import exponential_kernel, onedif_kernel
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import pickle
import random
import numpy as np
import networkx as nx
from src.weights import optimal_weight_matrix
from src.networks import generate_connected_erdos_renyi_graph
import math
from sklearn.gaussian_process.kernels import Matern
from src.generation import GPPSampleGenerator
# Assuming your class is defined in gppestimation.py
from src.estimation_torch import GPPEstimation
from scipy.optimize import minimize
import torch
import unittest
import sys
import time
import scipy.stats as stats
# Add the path where your Python packages are located


# %%


def construct(r, length_scale, nu):
    alpha = 1
    # length_scales=[0.3,0.1,0.03]
    # nu=0.5
    N = 10000
    mis_dis = 0.02
    l = math.sqrt(2*N)*mis_dis
    extent = -l/2, l/2, -l/2, l/2,
    coefficients = (-1, 2, 3, -2, 1)
    noise_level = 2
    J = 10
    con_pro = 0.5
    er = generate_connected_erdos_renyi_graph(J, con_pro)
    adj_matrix = nx.adjacency_matrix(er).todense()
    np.fill_diagonal(adj_matrix, 1)
    weights, _ = optimal_weight_matrix(adj_matrix=adj_matrix)
    weights = torch.tensor(weights, dtype=torch.double)

    # weights = torch.ones((J,J),dtype=torch.float64)/J

    kernel = alpha*Matern(length_scale=length_scale, nu=nu)
    sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=extent,
                                 kernel=kernel, coefficients=coefficients, noise=noise_level, seed=r)
    data, knots = sampler.generate_obs_gpp(m=100, method="random")
    dis_data = sampler.data_split(data, J)

    if nu == 0.5:
        gpp_estimation = GPPEstimation(
            dis_data, exponential_kernel, knots, weights)
    elif nu == 1.5:
        gpp_estimation = GPPEstimation(dis_data, onedif_kernel, knots, weights)
    else:
        raise ("incompleted")

    return gpp_estimation


nu_lengths = [(0.5, 0.033), (0.5, 0.1), (0.5, 0.234), 
              (1.5, 0.021 *math.sqrt(3)), 
              (1.5, 0.063*math.sqrt(3)), 
              (1.5, 0.148*math.sqrt(3))]
nu_length = nu_lengths[1]
nu = nu_length[0]
length_scale = nu_length[1]
N = 10000
m = 100

beta_true = torch.tensor([-1, 2, 3, -2, 1], dtype=torch.float64)
delta_true = torch.tensor(0.25, dtype=torch.float64).unsqueeze(0)
alpha_true = 1
theta_true = torch.tensor([alpha_true, length_scale], dtype=torch.float64)
parameters_true=torch.cat([beta_true,delta_true,theta_true])

# gpp_estimation = construct(0, length_scale, nu)
# beta = torch.tensor([-1, 2, 3, -2, 1], dtype=torch.float64)
# delta = torch.tensor(0.25, dtype=torch.float64)
# alpha = 1
# theta = torch.tensor([alpha, length_scale], dtype=torch.float64)
# V_matrices = gpp_estimation.ce_asy_variance_autodif(beta, delta, theta, -1)

# asy_variance_beta = torch.linalg.inv(V_matrices[0])/N
# asy_variance_delta = 1/V_matrices[1]/N
# asy_variance_theta = torch.linalg.inv(V_matrices[2])/m
# std_list = [torch.sqrt(torch.diag(asy_variance_beta)),
#             torch.sqrt(asy_variance_delta),
#             torch.sqrt(torch.diag(asy_variance_theta))]

with open(f'expriements/decentralized/varying_parameter/more_irregular/nu_{nu}_length_scale_{length_scale}_memeff.pkl', 'rb') as f:
    results = pickle.load(f)
error_rep = []
beta_list = []
delta_list = []
theta_list = []
std_list=[]
for r in range(100):
    print(r, end=" ")
    if type(results[r][1]) == str or type(results[r][1][1]) == str:
        error_rep.append(r)
        continue
    gpp_estimation = construct(r, length_scale, nu)

    # beta
    beta = results[r][1][2].squeeze()
    beta_list.append(beta)
    # delta
    delta = results[r][1][3].squeeze()
    delta_list.append(delta)
    # theta
    theta = results[r][1][4].squeeze()
    theta_list.append(theta)
    V_matrices = gpp_estimation.ce_asy_variance_autodif(beta, delta, theta, -1)
    asy_variance_beta = torch.linalg.inv(V_matrices[0])/N
    asy_variance_delta = 1/V_matrices[1]/N
    asy_variance_theta = torch.linalg.inv(V_matrices[2])/m
    std = torch.cat([
        torch.sqrt(torch.diag(asy_variance_beta)),
        torch.sqrt(torch.tensor([asy_variance_delta])),
        torch.sqrt(torch.diag(asy_variance_theta))
    ])
    std_list.append(std)

beta_tensor = torch.stack(beta_list)
delta_tensor = torch.stack(delta_list)
theta_tensor = torch.stack(theta_list)
std_tensor = torch.stack(std_list)

#empirical std
std_beta=torch.std(beta_tensor, dim=0, unbiased=True)
std_delta=torch.std(delta_tensor, unbiased=True).unsqueeze(0)
std_theta=torch.std(theta_tensor, dim=0, unbiased=True)
std_emp=torch.cat([std_beta,std_delta,std_theta])

#average of estimated std
valid_rows = ~torch.isnan(std_tensor).any(dim=1)

# Filter the tensor to exclude rows with NaN values
filtered_tensor = std_tensor[valid_rows]

std_est_avg=torch.mean(filtered_tensor,dim=0)



#confidence interval
alpha = 0.05  # Significance level (e.g., 95% confidence level)
z_alpha_half = stats.norm.ppf(1 - alpha / 2)  # Critical value
upper_bounds_list=[]
lower_bounds_list=[]
parameters_list=[]
is_in_list=[]
dim_para=len(parameters_true)
for r in range(100):
    beta=beta_tensor[r,:]
    delta=delta_tensor[r].unsqueeze(0)
    theta=theta_tensor[r,:]
    parameters=torch.cat([beta,delta,theta])
    stds=std_tensor[r,:]
    upper_bounds=parameters+z_alpha_half*stds
    lower_bounds=parameters-z_alpha_half*stds
    is_in=torch.zeros_like(parameters_true)
    for j in range(dim_para):
        if parameters_true[j]<=upper_bounds[j] and parameters_true[j]>lower_bounds[j]:
            is_in[j]=1
        else:
            is_in[j]=0
    parameters_list.append(parameters)
    upper_bounds_list.append(upper_bounds)
    lower_bounds_list.append(lower_bounds)
    is_in_list.append(is_in)
    
parameters_tensor=torch.stack(parameters_list)   
is_in_tensor=torch.stack(is_in_list)
upper_bounds_tensor=torch.stack(upper_bounds_list)
lower_bounds_tensor=torch.stack(lower_bounds_list)

cv_prob=torch.mean(is_in_tensor,dim=0)
results=[std_emp,std_est_avg,cv_prob,std_tensor,is_in_tensor,parameters_tensor,upper_bounds_tensor,lower_bounds_tensor]

with open("expriements/decentralized/CI/results.pkl", "wb") as file:
    pickle.dump(results, file)

