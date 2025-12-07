# %%

import torch
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt


# %%
#change the working directory
import os
os.chdir('/home/shij0d/Documents/Dis_Spatial/')


# %%
nu_lengths=[(0.5,0.033),(0.5,0.1),(0.5,0.234),(1.5,0.021),(1.5,0.063),(1.5,0.148)]
nu_lengths=[(1.5,0.021)]
beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
delta=torch.tensor(0.25,dtype=torch.float64)
alpha=1
J=10
scale=torch.tensor([1,math.sqrt(3)])

for nu_length in nu_lengths:
    nu=nu_length[0]
    length_scale=nu_length[1]
    theta=torch.tensor([alpha,length_scale],dtype=torch.float64)
    if nu==1.5:
        scale=torch.tensor([1,math.sqrt(3)])
    else:
        scale=torch.tensor([1.0,1.0])
    with open(f'expriements/decentralized/varying_parameter/more_irregular/nu_{nu}_length_scale_{length_scale}_memeff.pkl', 'rb') as f:
        results=pickle.load(f)
    param_rel_error=np.zeros(shape=(100,100))
    sta_error_beta=np.zeros(shape=(100,))
    sta_error_delta=np.zeros(shape=(100,))
    sta_error_theta=np.zeros(shape=(100,))
    error_global_rep=[]
    error_local_rep=[]
    error_dis_rep=[]
    for r in range(100):
        if type(results[r][1])==str:
            if results[r][1]=="local minimization error":
                error_local_rep.append(r)
            elif results[r][1]=="distributed minimization error":
                error_dis_rep.append(r)
        elif type(results[r][1][1])==str:
            error_global_rep.append(r)
    print(f"nu:{nu},length_scale:{length_scale},error_global_rep:{error_global_rep},error_local_rep:{error_local_rep},error_dis_rep:{error_dis_rep}")
    
        



# %%
