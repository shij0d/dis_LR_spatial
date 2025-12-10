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

results=[]
for nu_length in nu_lengths:
    nu = nu_length[0]
    length_scale = nu_length[1]
    print(f"nu:{nu},length_scale:{length_scale}")
    N = 10000
    m = 100

    beta_true = torch.tensor([-1, 2, 3, -2, 1], dtype=torch.float64)
    delta_true = torch.tensor(0.25, dtype=torch.float64).unsqueeze(0)
    alpha_true = 1
    theta_true = torch.tensor([alpha_true, length_scale], dtype=torch.float64)
    parameters_true=torch.cat([beta_true,delta_true,theta_true])
    delta_theta_true=torch.cat([delta_true,theta_true])

    min_eigenvalue_list=[]
    for r in range(100):
        print(r, end=" ")
        gpp_estimation = construct(r, length_scale, nu)
        Hessian = gpp_estimation.Hessian_delta_theta_Expected(delta_theta_true, delta_theta_true)
        #Hessian[0,0] divided by N, and the rest divided by m
        Hessian[0,0] = Hessian[0,0]/N
        Hessian[0,1:] = Hessian[0,1:]/m
        Hessian[1:,0] = Hessian[1:,0]/m
        Hessian[1:,1:] = Hessian[1:,1:]/m
        #compute the minimum eigenvalue of Hessian
        min_eigenvalue = torch.min(torch.linalg.eigvalsh(Hessian))
        min_eigenvalue_list.append(min_eigenvalue)
    min_eigenvalue_tensor = torch.stack(min_eigenvalue_list)
    results.append(min_eigenvalue_tensor)

with open("expriements/decentralized/HessianP/results_expected.pkl", "wb") as file:
        pickle.dump(results, file)


    # %%
