#%%
import torch
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel,onedif_kernel
from src.networks import generate_connected_erdos_renyi_graph
from src.weights import optimal_weight_matrix
from src.prediction import GPPPrediction
import networkx as nx
import numpy as np
import pickle
from functools import partial
from joblib import Parallel, delayed

#%%



def estimation_prediction(r,length_scale,nu):
    alpha=1
    #length_scales=[0.3,0.1,0.03]
    #nu=0.5
    N_est=10000
    N_pre=50
    N=N_est+N_pre
    mis_dis=0.02
    l=math.sqrt(2*N)*mis_dis
    extent=-l/2,l/2,-l/2,l/2,
    coefficients=(-1,2,3,-2,1)
    noise_level=math.sqrt(0.1)
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
    data_est=data[0:N_est]
    data_pre=data[N_est:]
    locations_pre=data_pre[:,0:2]
    z_pre=data_pre[:,2]
    X_pre=data_pre[:,3:]
    dis_data=sampler.data_split(data_est,J)
    
    if nu==0.5:
        kernelf=exponential_kernel
        
    elif nu==1.5:
        kernelf=onedif_kernel
    else:
        raise("incompleted")
    gpp_estimation = GPPEstimation(dis_data, kernelf, knots, weights)
        
    beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
    delta=torch.tensor(0.25,dtype=torch.float64)
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    x_true=gpp_estimation.argument2vector_lik(beta,delta,theta)
    try:
        

        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_true)
        gpp_prediction=GPPPrediction(locations_pre,kernelf,knots,X_pre,mu,Sigma,beta,delta,theta)
        optimal_pre=gpp_prediction.predict()
        
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        print("global optimization succeed")
        #print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    except Exception:
        optimal_estimator=(r, "global minimization error")
        print("global optimization failed")
  
    try:
        
        mu_list,Sigma_list,beta_list,delta_list,theta_list,_,_=gpp_estimation.get_local_minimizers_parallel(x_true,-1)

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
    #print(f"beta:{beta.squeeze().numpy()},delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
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
        de_pre_list=[]
        for j in range(J):
            mu=de_estimators[0][j]
            Sigma=de_estimators[1][j]
            beta=de_estimators[2][T-1][j]
            delta=de_estimators[3][T-1][j]
            theta=de_estimators[4][T-1][j]
            gpp_prediction=GPPPrediction(locations_pre,kernelf,knots,X_pre,mu,Sigma,beta,delta,theta)
            de_pre=gpp_prediction.predict()
            de_pre_list.append(de_pre)
        print("dis optimization succeed")
    except Exception:
        print("dis optimization failed")
        return (r, "distributed minimization error")
  
    return de_estimators,optimal_estimator,de_pre_list,optimal_pre,z_pre,locations_pre

#estimation_prediction(0,0.1,0.5)
nu_lengths=[(0.5,0.033),(0.5,0.1),(0.5,0.234),(1.5,0.021*math.sqrt(3)),(1.5,0.063*math.sqrt(3)),(1.5,0.148*math.sqrt(3))]
#nu_lengths=[(0.5,0.033),(1.5,0.021*math.sqrt(3))]
rs=[r for r in range(100)]
#rs=[8]
for nu_length in nu_lengths:
    nu=nu_length[0]
    
    length_scale=nu_length[1]
    if nu==1.5:
        length_scale_act=length_scale/math.sqrt(3)
    else:
        length_scale_act=length_scale
    print(f"nu:{nu},length_scale:{length_scale_act}")
    estimation_prediction_l=partial(estimation_prediction,length_scale=length_scale,nu=nu)
    
    results = [None] * len(rs)
    # Parallel execution for the list of rs, while maintaining the index (i)
    results = Parallel(n_jobs=-1)(
        delayed(lambda i, r: (i, estimation_prediction_l(r)))(i, r) for i, r in enumerate(rs)
    )
    # Assign results based on the index to maintain order
    for i, result in results:
        results[i] = result
    with open(f'expriements/decentralized/prediction/noise_sqrt(0.1)/nu_{nu}_length_scale_{length_scale_act}_memeff.pkl', 'wb') as f:
        pickle.dump(results, f)

    