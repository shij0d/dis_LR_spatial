#%%
import torch
from src.estimation_torch import GPPEstimation  
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import matern_kernel_factory
from src.networks import generate_connected_erdos_renyi_graph
from src.weights import optimal_weight_matrix
import networkx as nx
import numpy as np
import pickle
from joblib import Parallel, delayed

#%%



def estimate(r,nu):
    alpha=1
    length_scale=0.1
    nu_true=0.53
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

    kernel=alpha*Matern(length_scale=length_scale,nu=nu_true)
    sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level,seed=r)
    data,knots=sampler.generate_obs_gpp(m=100,method="random")
    dis_data=sampler.data_split(data,J)
    
    
    matern_kernel_nu=matern_kernel_factory(nu)
    gpp_estimation = GPPEstimation(dis_data, matern_kernel_nu, knots, weights)
    
        
    beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
    delta=torch.tensor(0.25,dtype=torch.float64)
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    x_true=gpp_estimation.argument2vector_lik(beta,delta,theta)
    try:
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
        optimal_estimator=(beta,delta,theta,result.fun)
        #print(optimal_estimator)
        print(f"r:{r},nu:{nu},global optimization succeed")
    except Exception:
        optimal_estimator=(r,nu, "global minimization error")
        print(f"r:{r},nu:{nu},global optimization failed")
    
    try:
        mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_true,job_num=-1)
        print(f"r:{r},nu:{nu},local optimization succeed")
    except Exception:
        print(f"r:{r},nu:{nu},local optimization failed")
        de_estimator=(r,nu, "local minimization error")
        
    if len(mu_list)==0:
        print(f"r:{r},nu:{nu},local optimization failed")
        de_estimator=(r,nu, "local minimization error")
        
    
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
        de_estimator=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=T,weights_round=6)
        de_estimator=(de_estimator[2],de_estimator[3],de_estimator[4],de_estimator[6])
        #print(de_estimator)
        #gpp_estimation.local_value()
        print(f"r:{r},nu:{nu},dis optimization succeed")
    except Exception:
        print(f"r:{r},nu:{nu},dis optimization failed")
        de_estimator=(r,nu, "dis minimization error")
        #estimator_de_op_list.append((de_estimator,optimal_estimator))
            
    return de_estimator,optimal_estimator


# run rs=[r for r in range(25)] and rs=[r for r in range(25,60)] in other machine to accelerate the computation
rs=[r for r in range(60,100)] 
nus=np.arange(0.2, 1, 0.1)
# Parallel execution for the list of rs, while maintaining the index (i)
results = Parallel(n_jobs=-1)(
    delayed(estimate)(r,nu) for  r in rs for nu in nus
)
with open(f'expriements/decentralized/estimating_nu/r_range(60,100)_nu_{0.53}.pkl', 'wb') as f:
    pickle.dump(results, f)