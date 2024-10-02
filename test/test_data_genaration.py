import sys


# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')

import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch_copy import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel
import numpy as np
import random
import pickle

#inital
# alpha=3
# length_scale=2
# nu=0.5
# kernel=alpha*Matern(length_scale=length_scale,nu=nu)
# N=10000
# mis_dis=0.2
# l=math.sqrt(2*N)
# extent=-l/2,l/2,-l/2,l/2,
# coefficients=(-1,2,3,-2,1)
# noise_level=2
# J=10
# sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level)
# data,knots=sampler.generate_obs_gpp(m=100,method="random")
# dis_data=sampler.data_split(data,J)
# weights = torch.ones((J,J),dtype=torch.float64)/J

# #save data for numeraical experienment
# data = {
#     "dis_data": dis_data,
#     "knots": knots,
#     "full_data": data
# }

# # Save variables to a file
# with open('./test/data.pkl', 'wb') as f:
#     pickle.dump(data, f)


# #get local minimizers which can be as initial points for optimization 
# gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
# beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
# delta=torch.tensor(0.25,dtype=torch.float64)
# theta=torch.tensor([3,2],dtype=torch.float64)
# x1=gpp_estimation.argument2vector_lik(beta,delta,theta)
# result1=gpp_estimation.get_minimier(x1)
# print('beta:',result1[2])
# print('delta:',result1[3])
# print('theta:',result1[4])
# print('convergence:',result1[5])
# mu_list,Sigma_list,beta_list,delta_list,theta_list,result_list=gpp_estimation.get_local_minimizers(x1)
# local_minizer_list = {
#     "mu_list": mu_list,
#     "Sigma_list": Sigma_list,
#     "beta_list": beta_list,
#     "delta_list": delta_list,
#     "theta_list": theta_list,
#     "result_list": result_list
# }

# # Save variables to a file
# with open('local_minizer_list.pkl', 'wb') as f:
#     pickle.dump(local_minizer_list, f)
    

#%% other settings
alpha=1
length_scales=[0.3]
nu=0.5
N=10000
mis_dis=0.01
#mis_dis=0.001
l=math.sqrt(2*N)*mis_dis
extent=-l/2,l/2,-l/2,l/2,
coefficients=(-1,2,3,-2,1)
noise_level=2
J=10
for length_scale in length_scales:
    kernel=alpha*Matern(length_scale=length_scale,nu=nu)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level)
    data,knots=sampler.generate_obs_gpp(m=100,method="random")
    dis_data=sampler.data_split(data,J)
    weights = torch.ones((J,J),dtype=torch.float64)/J
    #save data for numeraical experienment
    # data = {
    #     "dis_data": dis_data,
    #     "knots": knots,
    #     "full_data": data
    # }
    # with open(f'./test/data_length_scale_{length_scale}.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    #get local minimizers which can be as initial points for optimization 
    gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
    delta=torch.tensor(0.25,dtype=torch.float64)
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    #theta=torch.tensor([2,10],dtype=torch.float64)
    x_true=gpp_estimation.argument2vector_lik(beta,delta,theta)
    mu,Sigma,beta,delta,theta,result,hess=gpp_estimation.get_minimier(x_true)
    print(theta)
    hess=torch.tensor(hess)
    eigvals=torch.linalg.eigvalsh(hess)
    print(eigvals.numpy())
    print(theta[0]/theta[1])
    print(result)
    # global_minimizer={
    #     "mu": mu,
    #     "Sigma": Sigma,
    #     "beta": beta,
    #     "delta": delta,
    #     "theta": theta,
    #     "result": result
    # }
    # with open(f'./test/global_minizer_length_scale_{length_scale}.pkl', 'wb') as f:
    #     pickle.dump(global_minimizer, f)
    #mu,Sigma,beta,delta,theta,result=gpp_estimation.get_local_minimizer(4,x_true)
    mu_list,Sigma_list,beta_list,delta_list,theta_list,result_list=gpp_estimation.get_local_minimizers(x_true)
    local_minizer_list = {
        "mu_list": mu_list,
        "Sigma_list": Sigma_list,
        "beta_list": beta_list,
        "delta_list": delta_list,
        "theta_list": theta_list,
        "result_list": result_list
    }

    #Save variables to a file
    with open(f'./test/local_minizer_list_length_scale_{length_scale}.pkl', 'wb') as f:
        pickle.dump(local_minizer_list, f)
# %%
# with open('/home/shij0d/Documents/Dis_Spatial/test/local_minizer_list_length_scale_0.03.pkl', 'rb') as f:
#      local_minizer_list = pickle.load(f)
# for j in range(10):
#     print(local_minizer_list['theta_list'][j])
#     print(local_minizer_list['result_list'][j])
# %%

#%%
