import sys


# Add the path where your Python packages are located
sys.path.append('/home/shij0d/Documents/Dis_Spatial')
import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch import GPPEstimation 
#from src.estimation_torch_copy import GPPEstimation  # Assuming your class is defined in gppestimation.py
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
with open('./test/data.pkl', 'rb') as f:
    data = pickle.load(f)
dis_data=data['dis_data']
knots=data['knots']
full_data=data['full_data']
J=len(dis_data)
with open('./test/local_minizer_list.pkl', 'rb') as f:
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

#initalizat the model with full connected network
weights_full = torch.ones((J,J),dtype=torch.float64)/J #
gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights_full)


# Other different networks

# er = generate_connected_erdos_renyi_graph(J, 0.8)
# adj_matrix=nx.adjacency_matrix(er).todense()
# np.fill_diagonal(adj_matrix, 1)
# weights_er_p_dot8,_=optimal_weight_matrix(adj_matrix=adj_matrix)
# weights_er_p_dot8=torch.tensor(weights_er_p_dot8,dtype=torch.double)

# #result
# T=50
# gpp_estimation.weights=weights_er_p_dot8
# result_er_p_dot8=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T)



er = generate_connected_erdos_renyi_graph(J, 0.5)
adj_matrix=nx.adjacency_matrix(er).todense()
np.fill_diagonal(adj_matrix, 1)
weights_er_p_dot5,_=optimal_weight_matrix(adj_matrix=adj_matrix)
weights_er_p_dot5=torch.tensor(weights_er_p_dot5,dtype=torch.double)
#result
T=100
gpp_estimation.weights=weights_er_p_dot5
result_er_p_dot5=gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T)


#result_er_p_dot5=gpp_estimation.ce_optimize_stage2(mu,Sigma,beta,delta,theta,T)


import pickle

# Load the data back
with open('saved_tensors.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Access the loaded tensors
mu = loaded_data['mu']
Sigma = loaded_data['Sigma']
beta = loaded_data['beta']
delta = loaded_data['delta']
theta = loaded_data['theta']


with open('saved_tensors_copy.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Access the loaded tensors
mu_copy = loaded_data['mu']
Sigma_copy = loaded_data['Sigma']
beta_copy = loaded_data['beta']
delta_copy = loaded_data['delta']
theta_copy = loaded_data['theta']
a=1