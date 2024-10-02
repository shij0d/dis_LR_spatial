#%%
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

#%%



def estimate(r,N):
    alpha=1
    length_scale=0.1
    nu=0.5
    #N=10000
    mis_dis=0.01
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
    try:
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        print("global optimization succeed")
    except Exception:
        optimal_estimator=(r, "global minimization error")
        print("global optimization failed")
    
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
        de_estimators=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T)
        print("dis optimization succeed")
    except Exception:
        print("dis optimization failed")
        return (r, "distributed minimization error")
  
    return de_estimators,optimal_estimator

num_cores = multiprocessing.cpu_count()
Ns=[20000,40000]
#nu_lengths=[(1.5,0.148)]
rs=[r for r in range(100)]
for N in Ns:
    print(f"N:{N}")
    estimate_l=partial(estimate,N=N)
    results=[]
    for r in rs:
        print(f"r:{r}")
        result=estimate_l(r)
        results.append(result)
    with open(f'/home/shij0d/Documents/Dis_Spatial/expriements/decentralized/varying_rank/N_{N}.pkl', 'wb') as f:
        pickle.dump(results, f)
    # with multiprocessing.Pool(processes=num_cores//2-1) as pool:
    #     results = pool.map(estimate_l,rs)
    #     with open(f'/home/shij0d/Documents/Dis_Spatial/expriements/decentralized/varying_parameter/nu_{nu}_length_scale_{length_scale}.pkl', 'wb') as f:
    #         pickle.dump(results, f)

#%% illustrate the result
# length_scales=[0.3,0.1,0.03]
# beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
# delta=torch.tensor(0.25,dtype=torch.float64)
# alpha=1
# for length_scale in length_scales:
#     theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
#     with open(f'/home/shij0d/Documents/Dis_Spatial/expriements/centralized/exp1_parameter_length_scale_{length_scale}.pkl', 'rb') as f:
#         results=pickle.load(f)
#     param_rel_error=np.zeros(shape=(100,50))
#     for r in range(100):
#         for t in range(50):
#             rel_dif_beta=torch.abs(results[r][0][2][t]-results[r][1][2]).squeeze()/torch.abs(beta)
#             rel_dif_delta=torch.abs(results[r][0][3][t]-results[r][1][3]).squeeze()/torch.abs(delta)
#             rel_dif_theta=torch.abs(results[r][0][4][t]-results[r][1][4]).squeeze()/torch.abs(theta)
#             param_rel_error[r,t]=torch.sqrt(torch.square(torch.norm(rel_dif_beta))+torch.square(torch.norm(rel_dif_delta))+torch.square(torch.norm(rel_dif_theta))).numpy()
#     # Calculate the mean, standard deviation, max, and min across replications for each iteration
#     param_rel_error=np.log10(param_rel_error)
#     mean_rel_error= np.mean(param_rel_error, axis=0)
#     std_rel_error = np.std(param_rel_error, axis=0)
#     percentile_10_error = np.percentile(param_rel_error, 10, axis=0)
#     percentile_90_error = np.percentile(param_rel_error, 90, axis=0)
#     max_rel_error = np.max(param_rel_error, axis=0)
#     min_rel_error = np.min(param_rel_error, axis=0)
    

#     # Plot the mean convergence curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(mean_rel_error, label='Mean Relative Error', color='blue')

#     # Plot the max and min convergence curves
#     plt.plot(max_rel_error, label='Max Relative Error', color='red', linestyle='--')
#     plt.plot(min_rel_error, label='Min Relative Error', color='green', linestyle='--')

#     # Fill between the 25th and 75th percentiles
#     plt.fill_between(range(50), percentile_10_error, percentile_90_error, color='blue', alpha=0.3, label='10th-90th Percentile Error')

#     # Customize the plot
#     plt.title(f'Length Scale:{length_scale}')
#     plt.xlabel('Iteration')
#     plt.ylabel('Logarithmic Error')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'exp1_length_scale:{length_scale}.pdf', dpi=300)  
#     # Show the plot
#     plt.show()


# %%
