#path_project="/home/shij0d/Documents/Dis_Spatial"
#import sys
#sys.path.append(path_project)
import warnings
# Suppress FutureWarning from torch.load
#warnings.filterwarnings("ignore")
warnings.simplefilter("error", FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.optimize import minimize
from src.estimation_torch_real_data import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.prediction import GPPPrediction

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
path='real_data/Second_scenario/MRA_codeAndData/MIRSmra.csv'
full_data=pd.read_csv(path,header=None,index_col=None).values

full_data[:,0]=full_data[:,0]-180
full_data[:, [0, 1]] = full_data[:, [1, 0]]

np.random.seed(SEED)  # Set seed for reproducibility
full_data = np.random.permutation(full_data) 
locations=full_data[:,:2]

beta=np.mean(full_data[:,2])
full_data[:, 2] -= beta
N=full_data.shape[0]

#divide to data into two parts for estimation and prediction
N_est=N-1000
data_est=full_data[:N_est,:] #estimation
data_pre=full_data[N_est:,:]
locations_est=data_est[:,:2]
locations_est_list = [tuple(row) for row in locations_est]

locations_pre=data_pre[:,:2]
y_true=data_pre[:,2]
file_path_save='real_data/Second_scenario/RMSPE_varying_m/result_prediction_y_true.pkl'
with open(file_path_save, "wb") as file:
        pickle.dump(y_true, file)
J=16
dis_data = np.array_split(data_est, J)

def generate_knots(m,locations):
    lat_min,lat_max=np.min(locations[:,0]),np.max(locations[:,0])
    lon_min,lon_max=np.min(locations[:,1]),np.max(locations[:,1])
    ratio=(lat_max-lat_min)/(lon_max-lon_min)
    num_divisions_lon = int(np.ceil(np.sqrt(m / ratio)))
    num_divisions_lat = int(np.ceil(m / num_divisions_lon))
    lats=np.linspace(lat_min, lat_max,num_divisions_lat)
    lons=np.linspace(lon_min, lon_max,num_divisions_lon)
    knots = [(lat, lon) for lat in lats for lon in lons]
    if len(knots)>m:
        random.seed(SEED)
        knots = random.sample(knots, m)
    return knots

def initial(nu,knots):
    
    N=500
    J=1
    nu=1.5
    data=data_est[:N,:]
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
def estimation_prediction(m,nu=1.5):
    J=len(dis_data)
    con_pro=0.5
    er = generate_connected_erdos_renyi_graph(J, con_pro)
    adj_matrix=nx.adjacency_matrix(er).todense()
    np.fill_diagonal(adj_matrix, 1)
    weights,_=optimal_weight_matrix(adj_matrix=adj_matrix)
    weights=torch.tensor(weights,dtype=torch.double)
    
    #grid knots
    knots=generate_knots(m,locations)
    
    # #random knots
    # random.seed(SEED)
    # knots = random.sample(locations_est_list, m)

    #initial value
    #random.seed(SEED)
    #knots_inital = random.sample(locations_est_list, 60)
    initial_m=60
    knots_inital = generate_knots(initial_m, locations)
    initial_estimator=initial(nu,knots_inital)
    
    
    if nu==0.5:
        kernelf=partial(exponential_kernel,type= "chordal")   
    elif nu==1.5:
        kernelf=partial(onedif_kernel,type="chordal")    
    else:
        kernelf= partial(matern_kernel,nu=nu,type="chordal")  
    gpp_estimation = GPPEstimation(dis_data,kernelf, knots, weights)
    
    ##initial value
    # beta_ini=None
    # delta_ini=torch.tensor(1,dtype=torch.float64)
    # alpha_ini=10
    # length_scale_ini=0.5
    # theta_ini=torch.tensor([alpha_ini,length_scale_ini],dtype=torch.float64)
    # x_ini=gpp_estimation.argument2vector_lik(beta_ini,delta_ini,theta_ini)
    x_ini=gpp_estimation.argument2vector_lik(initial_estimator[2],initial_estimator[3],initial_estimator[4])
    
    try:
        mu,Sigma,beta,delta,theta,result=gpp_estimation.get_minimier(x_ini)
        gpp_prediction=GPPPrediction(locations_pre,kernelf,knots,None,mu,Sigma,None,delta,theta)
        optimal_pre=gpp_prediction.predict()
        optimal_estimator=(mu,Sigma,beta,delta,theta,result)
        print("global optimization succeed")
        print(f"delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    except Exception:
        optimal_estimator=("global minimization error")
        print("global optimization failed")
        optimal_pre=None
    
    # try:
    #     T=32
    #     ce_estimators=gpp_estimation.ce_optimize_stage2(initial_estimator[0],initial_estimator[1],initial_estimator[2],initial_estimator[3],initial_estimator[4],T,J,thread_num=None,backend='threading') 
    #     mu,Sigma,beta_list,delta_list,theta_list,_=ce_estimators
    #     delta=delta_list[-1]
    #     theta=theta_list[-1]
    #     gpp_prediction=GPPPrediction(locations_pre,kernelf,knots,None,mu,Sigma,None,delta,theta)
    #     distributed_pre=gpp_prediction.predict()
    #     print("distributed optimization succeed")
    #     print(f"delta:{delta.numpy()},theta:{theta.squeeze().numpy()}")
    # except Exception:
    #     ce_estimators=("distributed minimization error")
    #     print("distributed optimization failed")
    #     distributed_pre=None
    
    
    return optimal_estimator[2:],optimal_pre#,ce_estimators[2:],distributed_pre

ms=[400,500]
results=[]
file_path_save='real_data/Second_scenario/RMSPE_varying_m/result_prediction_varying_m_grid_knots_more.pkl'
for i,m in enumerate(ms):
    print(m)
    result_i=estimation_prediction(m)
    results.append(result_i)
    with open(file_path_save, "wb") as file:
        pickle.dump(results, file)
