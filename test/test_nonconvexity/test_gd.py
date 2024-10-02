import sys


# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')

import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from src.weights import optimal_weight_matrix
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel
import numpy as np
import random
import pickle
import networkx as nx
from src.networks import generate_connected_erdos_renyi_graph

#load data and local minizers
length_scales=[0.3,0.1,0.03]
length_scales=[0.3]
for length_scale in length_scales:
    with open(f'/home/shij0d/Documents/Dis_Spatial/test/data_length_scale_{length_scale}.pkl', 'rb') as f:
        data = pickle.load(f)
    dis_data=data['dis_data']
    knots=data['knots']
    full_data=data['full_data']
    J=len(dis_data)
    with open(f'/home/shij0d/Documents/Dis_Spatial/test/local_minizer_list_length_scale_{length_scale}.pkl', 'rb') as f:
        local_minizer_list = pickle.load(f)

    # Extract individual variables
    mu_list = local_minizer_list["mu_list"]
    Sigma_list = local_minizer_list["Sigma_list"]
    beta_list = local_minizer_list["beta_list"]
    delta_list = local_minizer_list["delta_list"]
    theta_list = local_minizer_list["theta_list"]
    result_list = local_minizer_list["result_list"]

    mu=mu_list[0]
    Sigma=Sigma_list[0]
    beta=beta_list[0]
    delta=delta_list[0]
    theta=theta_list[0]

    for j in range(1,J):
        mu+=mu_list[j]
        Sigma+=Sigma_list[j]
        beta+=beta_list[j]
        delta+=delta_list[j]
        theta+=theta_list[j]
    mu=mu/J
    Sigma=Sigma/J
    beta=beta/J
    delta=delta/J
    theta=theta/J

    for j in range(J):
        mu_list[j]=mu
        Sigma_list[j]=Sigma
        beta_list[j]=beta
        delta_list[j]=delta
        theta_list[j]=theta
    

    with open(f'/home/shij0d/Documents/Dis_Spatial/test/global_minizer_length_scale_{length_scale}.pkl', 'rb') as f:
        global_minizer = pickle.load(f)
    mu = global_minizer["mu"]
    Sigma = global_minizer["Sigma"]
    beta = global_minizer["beta"]
    delta = global_minizer["delta"]
    theta = global_minizer["theta"].clone()
    theta[0]=theta[0]*10
    theta[1]=theta[1]*2
    for j in range(J):
        mu_list[j]=mu
        Sigma_list[j]=Sigma
        beta_list[j]=beta
        delta_list[j]=delta
        theta_list[j]=theta
    #initalizat the model with full connected network
    T=50
    weights_full = torch.ones((J,J),dtype=torch.float64)/J #
    gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights_full)
    initial_params=gpp_estimation.argument2vector(mu,Sigma,beta,delta,theta)
    result_stage1=gpp_estimation.dis_opimize_stage1(initial_params,T=10000,noisy=False)
    mu,Sigma,beta,delta,theta=gpp_estimation.vector2arguments(result_stage1[-1])
    
    theta_stage1=[]
    for t in range(len(result_stage1)):
        _,_,_,_,theta=gpp_estimation.vector2arguments(result_stage1[t])
        theta_stage1.append(theta)
    with open(f'/home/shij0d/Documents/Dis_Spatial/test/test_nonconvexity/test_gd_{length_scale}.pkl', 'wb') as f:
        pickle.dump(theta_stage1, f)
