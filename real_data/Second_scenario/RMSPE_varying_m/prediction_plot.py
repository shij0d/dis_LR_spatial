#%%
#import sys

#path_project="/home/shij0d/Documents/Dis_Spatial"
## Add the path where your Python packages are located
#sys.path.append(path_project)

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
import os

path_results='real_data/Second_scenario/RMSPE_varying_m/result_prediction_varying_m_grid_knots.pkl'
path_results_more='real_data/Second_scenario/RMSPE_varying_m/result_prediction_varying_m_grid_knots_more.pkl'
path_true='real_data/Second_scenario/RMSPE_varying_m/result_prediction_y_true.pkl'

with open(path_results, 'rb') as f:
    results=pickle.load(f)
with open(path_results_more, 'rb') as f:
    results_more=pickle.load(f)
with open(path_true, 'rb') as f:
    y_true=pickle.load(f)

M=len(results)
RMSPE=np.zeros((M,))
for m in range(M):
    if results[m][1]!=None:
        y_pre=results[m][1][0].reshape((-1,))
        RMSPE[m]=math.sqrt(torch.mean((y_pre-y_true)**2))
M=len(results_more)
RMSPE_more=np.zeros((M,))
for m in range(M):
    if results_more[m][1]!=None:
        y_pre=results_more[m][1][0].reshape((-1,))
        RMSPE_more[m]=math.sqrt(torch.mean((y_pre-y_true)**2))
#RMSPE=np.concatenate((RMSPE,RMSPE_more))
#%%   

ms=[60,70,80,90,100,150,200,250,300,400,500]
ms=[60,80,100,150,200,250]
RMSPE=RMSPE[[0,2,4,5,6,7]]
# Plot MLE and DE bars with colors and transparency
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ms, RMSPE, 'o--', label='RMSPE', color='salmon', markersize=6)
x_ticks=ms
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=10)
ax.tick_params(axis='y', labelsize=14)

# Customize plot
ax.set_xlabel('Rank (m)',fontsize=14)
ax.set_ylabel('RMSPE',fontsize=14)
ax.legend(fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

plt.savefig("real_data/RMSPE_varying_m.pdf")
plt.show()
# %%
