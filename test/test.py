
from test.test_estimation_torch import TestGPPEstimation
from src.kernel import exponential_kernel
import numpy as np
import torch


import pickle

with open('variables.pkl', 'rb') as f:
    loaded_variables = pickle.load(f)

# Extract individual variables
mu_list = loaded_variables["mu_list"]
Sigma_list = loaded_variables["Sigma_list"]
beta_list = loaded_variables["beta_list"]
delta_list = loaded_variables["delta_list"]
theta_list = loaded_variables["theta_list"]
result_list = loaded_variables["result_list"]
J=len(mu_list)
mu=mu_list[0]
Sigma=Sigma_list[0]
theta=theta_list[0]
for j in range(1,J):
    mu=mu+mu_list[j]
    Sigma=Sigma+Sigma_list[j]
    theta=theta+theta_list[j]
mu=mu/J
Sigma=Sigma/J
theta=theta/J


testGPPE=TestGPPEstimation()
testGPPE.setUp()
knots=testGPPE.gpp_estimation.knots
dis_data=testGPPE.gpp_estimation.dis_data
kernel=exponential_kernel

def com_fun(mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):
    K = kernel(knots,knots, theta)
    invK = torch.inverse(K)
    f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))
    return f_value
    
def com_fun_grad(mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):
    theta=theta.clone()
    if theta.grad is not None:
        theta.grad.data.zero_()
    theta.requires_grad_(True)
    K = kernel(knots,knots, theta)
    invK = torch.inverse(K)
    f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))
    f_value.backward()
    grad = theta.grad
    return grad

def sum_local_fun(mu:torch.Tensor,Sigma:torch.Tensor,beta:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor):
    return



theta_ite_list=[theta]
T=50
step_size=0.1
grad_list=[]
for t in range(T):
    
    grad=com_fun_grad(theta)
    theta=theta-step_size*grad
    theta_ite_list.append(theta)
    grad_list.append(grad)
