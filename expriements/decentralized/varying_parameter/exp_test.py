#%%
import sys
import time

# Add the path where your Python packages are located
#sys.path.append('/home/shij0d/Documents/Dis_Spatial')

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

#beta:[-1.01605587  2.01384319  3.01852677 -2.04447403  1.01076155],delta:0.2530022707264758,theta:[1.16931021 0.10380016]

def estimate(r,length_scale,nu):
    alpha=1
    #length_scales=[0.3,0.1,0.03]
    #nu=0.5
    N=10000
    mis_dis=0.01
    l=math.sqrt(2*N)*mis_dis*1.5
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
        print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
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
    print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
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
  
    return optimal_estimator,de_estimators


num_cores = multiprocessing.cpu_count()
nu_lengths=[(0.5,0.033),(0.5,0.1),(0.5,0.234),(1.5,0.021*math.sqrt(3)),(1.5,0.063*math.sqrt(3)),(1.5,0.148*math.sqrt(3))]
nu_lengths=[nu_lengths[1]]
rs=[r for r in range(100)]
rs=[8]
for nu_length in nu_lengths:
    nu=nu_length[0]
    
    length_scale=nu_length[1]
    if nu==1.5:
        length_scale_act=length_scale/math.sqrt(3)
    else:
        length_scale_act=length_scale
    print(f"nu:{nu},length_scale:{length_scale_act}")
    estimate_l=partial(estimate,length_scale=length_scale,nu=nu)
    results=[]
    for r in rs:
        print(f"r:{r}")
        result=estimate_l(r)
        results.append(result)
    # with open(f'expriements/decentralized/varying_parameter/mindis_0.01/temp.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    # results = [None] * len(rs)
    # # Parallel execution for the list of rs, while maintaining the index (i)
    # results = Parallel(n_jobs=-1)(
    #     delayed(lambda i, r: (i, estimate_l(r)))(i, r) for i, r in enumerate(rs)
    # )
    # # Assign results based on the index to maintain order
    # for i, result in results:
    #     results[i] = result
    # with open(f'expriements/decentralized/varying_parameter/mindis_0.01_irregular/nu_{nu}_length_scale_{length_scale_act}_weights_round_{6}.pkl', 'wb') as f:
    #     pickle.dump(results, f)

   
# %%
r=0
J=10
beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
delta=torch.tensor(0.25,dtype=torch.float64)
theta=torch.tensor([0.5,0.1],dtype=torch.float64)
param_rel_error=np.zeros(shape=(100,))

for t in range(100):
    rel_error=torch.zeros((J,))
    for j in range(J):
        rel_dif_beta=torch.abs(results[r][1][2][t][j]-results[r][0][2]).squeeze()/torch.abs(beta)
        rel_dif_delta=torch.abs(results[r][1][3][t][j]-results[r][0][3]).squeeze()/torch.abs(delta)
        rel_dif_theta=torch.abs(results[r][1][4][t][j]-results[r][0][4]).squeeze()/torch.abs(theta)
        rel_error[j]=torch.sqrt(torch.square(torch.norm(rel_dif_beta))+torch.square(torch.norm(rel_dif_delta))+torch.square(torch.norm(rel_dif_theta)))
    param_rel_error[t]=rel_error.max().numpy()
param_rel_error=np.log10(param_rel_error)
plt.plot(param_rel_error)
plt.title("Parameter Relative Error for 9th Row")
plt.xlabel("Index")
plt.ylabel("Relative Error")
plt.grid(True)
plt.show()


# %%
