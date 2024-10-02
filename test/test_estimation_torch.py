import unittest
import torch
from scipy.optimize import minimize
from src.estimation_torch import GPPEstimation  # Assuming your class is defined in gppestimation.py
from src.generation import GPPSampleGenerator
from sklearn.gaussian_process.kernels import Matern
import math
from src.kernel import exponential_kernel
import numpy as np
from scipy.optimize import approx_fprime
import random


class TestGPPEstimation(unittest.TestCase):
    
    def setUp(self):
        
        alpha=3
        length_scale=2
        nu=0.5
        kernel=alpha*Matern(length_scale=length_scale,nu=nu)
        N=10000
        mis_dis=0.2
        l=math.sqrt(2*N)
        extent=-l/2,l/2,-l/2,l/2,
        coefficients=(-1,2,3,-2,1)
        noise_level=2
        J=10
        sampler=GPPSampleGenerator(num=N,min_dis=mis_dis,extent=extent,kernel=kernel,coefficients=coefficients,noise=noise_level)
        data,knots=sampler.generate_obs_gpp(m=100,method="random")
        dis_data=sampler.data_split(data,J)

        weights = torch.ones((J,J),dtype=torch.float64)/J
        
        self.gpp_estimation = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    
    def test_vector2arguments(self):
        params = torch.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        mu, Sigma, beta, delta, theta = self.gpp_estimation.vector2arguments(params)
        self.assertEqual(mu.shape, (self.gpp_estimation.m, 1))
        self.assertEqual(Sigma.shape, (self.gpp_estimation.m, self.gpp_estimation.m))
        self.assertEqual(beta.shape, (self.gpp_estimation.p, 1))
        self.assertEqual(delta.shape, ())
        self.assertEqual(theta.shape, (2, 1))

    def test_argument2vector(self):
        mu = torch.randn(self.gpp_estimation.m, 1,dtype=torch.float64)
        Sigma = torch.eye(self.gpp_estimation.m,dtype=torch.float64)
        beta = torch.randn(self.gpp_estimation.p, 1,dtype=torch.float64)
        delta = torch.tensor(1.0,dtype=torch.float64)
        theta = torch.tensor([[1.0], [2.0]],dtype=torch.float64)
        params = self.gpp_estimation.argument2vector(mu, Sigma, beta, delta, theta)
        self.assertEqual(params.shape, (self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2,))

    def test_vector2arguments_lik(self):
        params = torch.randn(self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        beta, delta, theta = self.gpp_estimation.vector2arguments_lik(params)
        self.assertEqual(beta.shape, (self.gpp_estimation.p, 1))
        self.assertEqual(delta.shape, ())
        self.assertEqual(theta.shape, (2, 1))

    def test_argument2vector_lik(self):
        beta = torch.randn(self.gpp_estimation.p, 1,dtype=torch.float64)
        delta = torch.tensor(1.0,dtype=torch.float64)
        theta = torch.tensor([[1.0], [2.0]],dtype=torch.float64)
        params = self.gpp_estimation.argument2vector_lik(beta, delta, theta)
        self.assertEqual(params.shape, (self.gpp_estimation.p + 1 + 2,))

    def test_local_fun(self):
        data = self.gpp_estimation.dis_data[0]
        local_locs = data[:, :2]
        local_z = data[:, 2].reshape(-1,1)
        local_X = data[:, 3:]
        params = torch.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        f_value, grad = self.gpp_estimation.local_fun(local_locs, local_z, local_X, params, requires_grad=True)
        self.assertIsInstance(f_value, torch.Tensor)
        self.assertIsInstance(grad, torch.Tensor)
    
    # def test_grad(self):
    #     data = self.gpp_estimation.dis_data[0]
    #     local_locs = data[:, :2]
    #     local_z = data[:, 2].reshape(-1,1)
    #     local_X = data[:, 3:]
    #     params = torch.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2,requires_grad=True)
    #     test = gradcheck(self.gpp_estimation.local_fun, (local_locs, local_z, local_X, params), eps=1e-6, atol=1e-4)
    #     print("Gradients check passed:", test)

    def test_local_fun_wrapper(self):
        fun, gradf = self.gpp_estimation.local_fun_wrapper(0, requires_grad=True)
        params =np.random.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2)
        self.assertIsInstance(fun(params), np.ndarray)
        self.assertIsInstance(gradf(params), np.ndarray)

    def check_local_grad(self):
        fun, gradf = self.gpp_estimation.local_fun_wrapper(0, requires_grad=True)
        params = np.random.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2)
        grad_auto=gradf(params)
        grad_fini=approx_fprime(params,fun)
        result=torch.allclose(torch.tensor(grad_auto),torch.tensor(grad_fini),rtol=1e-04, atol=1e-02,)
        return result
    

    def test_com_fun(self):
        params = torch.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        f_value, grad = self.gpp_estimation.com_fun(params, requires_grad=True)
        self.assertIsInstance(f_value, torch.Tensor)
        self.assertIsInstance(grad, torch.Tensor)

    def test_com_fun_wrapper(self):
        fun, gradf = self.gpp_estimation.com_fun_wrapper(requires_grad=True)
        params = np.random.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2)
        self.assertIsInstance(fun(params), np.ndarray)
        self.assertIsInstance(gradf(params), np.ndarray)

    def check_com_grad(self):
        fun, gradf = self.gpp_estimation.com_fun_wrapper(requires_grad=True)
        params = np.random.randn(self.gpp_estimation.m + self.gpp_estimation.m * (self.gpp_estimation.m + 1) // 2 + self.gpp_estimation.p + 1 + 2)
        grad_auto=gradf(params)
        grad_fini=approx_fprime(params,fun)
        result=torch.allclose(torch.tensor(grad_auto),torch.tensor(grad_fini),rtol=1e-04, atol=1e-02,)
        return result
    
    def test_neg_log_lik(self):
        data = self.gpp_estimation.dis_data[0]
        local_locs = data[:, :2]
        local_z = data[:, 2].unsqueeze(1)
        local_X = data[:, 3:]
        random.seed(2024)
        params = torch.randn(self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
        delta=torch.tensor(0.25,dtype=torch.float64)
        theta=torch.tensor([3,2],dtype=torch.float64)
        params=self.gpp_estimation.argument2vector_lik(beta,delta,theta)
        f_value, grad = self.gpp_estimation.neg_log_lik(local_locs, local_z, local_X, params, requires_grad=True)
        self.assertIsInstance(f_value, torch.Tensor)
        self.assertIsInstance(grad, torch.Tensor)

    def test_local_neg_log_lik_wrapper(self):
        fun, gradf = self.gpp_estimation.local_neg_log_lik_wrapper(0, requires_grad=True)
        params = np.random.randn(self.gpp_estimation.p + 1 + 2)
        self.assertIsInstance(fun(params), np.ndarray)
        self.assertIsInstance(gradf(params), np.ndarray)

    def check_local_neg_log_lik_grad(self):
        fun, gradf = self.gpp_estimation.local_neg_log_lik_wrapper(0, requires_grad=True)
        params = np.random.randn(self.gpp_estimation.p + 1 + 2)
        grad_auto=gradf(params)
        grad_fini=approx_fprime(params,fun)
        result=torch.allclose(torch.tensor(grad_auto),torch.tensor(grad_fini),rtol=1e-04, atol=1e-02,)
        return result
    
    def test_neg_log_lik_wrapper(self):
        fun, gradf = self.gpp_estimation.neg_log_lik_wrapper(requires_grad=True)
        params = np.random.randn(self.gpp_estimation.p + 1 + 2)
        self.assertIsInstance(fun(params), np.ndarray)
        self.assertIsInstance(gradf(params), np.ndarray)

    def check_neg_log_lik_grad(self):
        fun, gradf = self.gpp_estimation.neg_log_lik_wrapper(requires_grad=True)
        params = np.random.randn(self.gpp_estimation.p + 1 + 2)
        grad_auto=gradf(params)
        grad_fini=approx_fprime(params,fun)
        result=torch.allclose(torch.tensor(grad_auto),torch.tensor(grad_fini),rtol=1e-04, atol=1e-02,)
        return result
    
    def test_get_local_pos(self):
        params = torch.randn(self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        mu, Sigma = self.gpp_estimation.get_local_pos(0, params)
        self.assertIsInstance(mu, torch.Tensor)
        self.assertIsInstance(Sigma, torch.Tensor)

    def test_get_pos(self):
        params = torch.randn(self.gpp_estimation.p + 1 + 2,dtype=torch.float64)
        mu, Sigma = self.gpp_estimation.get_pos(params)
        self.assertIsInstance(mu, torch.Tensor)
        self.assertIsInstance(Sigma, torch.Tensor)
        
        # Additional checks if necessary
        
    # Add unit tests for other functions if needed
    
# if __name__ == '__main__':
#     unittest.main()


testGPPE=TestGPPEstimation()
testGPPE.setUp()
# testGPPE.test_vector2arguments()
# testGPPE.test_argument2vector()
# testGPPE.test_vector2arguments_lik()
# testGPPE.test_argument2vector_lik()
# testGPPE.test_local_fun()
# testGPPE.test_local_fun_wrapper()
# testGPPE.test_com_fun()
# testGPPE.test_com_fun_wrapper()
# testGPPE.test_neg_log_lik()

# testGPPE.test_local_neg_log_lik_wrapper()
# testGPPE.test_neg_log_lik_wrapper()
# testGPPE.test_get_local_pos()
# testGPPE.test_get_pos()

# check1=testGPPE.check_local_grad()
# check2=testGPPE.check_com_grad()
# check3=testGPPE.check_local_neg_log_lik_grad()
# check4=testGPPE.check_neg_log_lik_grad()

# print("check_local_grad:", check1)
# print("check_com_grad:", check2)
# print("check_local_neg_log_lik_grad:", check3)
# print("check_neg_log_lik_grad:", check4)

# np.random.seed(2023) 
# x0=np.random.randn(testGPPE.gpp_estimation.p + 1 + 2)

# result0=testGPPE.gpp_estimation.get_local_minimizer(0,x0)
# print('beta:',result0[2])
# print('delta:',result0[3])
# print('theta:',result0[4])
# print('convergence:',result0[5])

# beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
# delta=torch.tensor(0.25,dtype=torch.float64)
# theta=torch.tensor([3,2],dtype=torch.float64)
# x1=testGPPE.gpp_estimation.argument2vector_lik(beta,delta,theta)
# result1=testGPPE.gpp_estimation.get_local_minimizer(0,x1)
# print('beta:',result1[2])
# print('delta:',result1[3])
# print('theta:',result1[4])
# print('convergence:',result1[5])

# beta=torch.tensor([-1,2,3,-2,1],dtype=torch.float64)
# delta=torch.tensor(0.25,dtype=torch.float64)
# theta=torch.tensor([3,2],dtype=torch.float64)
# x1=testGPPE.gpp_estimation.argument2vector_lik(beta,delta,theta)
# result1=testGPPE.gpp_estimation.get_minimier(x1)
# mu=result1[0]
# Sigma=result1[1]
# beta=result1[2]
# delta=result1[3]
# theta=result1[4]

# print('beta:',result1[2])
# print('delta:',result1[3])
# print('theta:',result1[4])
# print('convergence:',result1[5])

# mu_list,Sigma_list,beta_list,delta_list,theta_list,result_list=testGPPE.gpp_estimation.get_local_minimizers(x1)
# variables = {
#     "mu_list": mu_list,
#     "Sigma_list": Sigma_list,
#     "beta_list": beta_list,
#     "delta_list": delta_list,
#     "theta_list": theta_list,
#     "result_list": result_list
# }
# import pickle

# # Save variables to a file
# with open('variables.pkl', 'wb') as f:
#     pickle.dump(variables, f)

#print("Variables saved successfully.")
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
# J=len(mu_list)
# for j in range(J):
#     mu_list[j]=result1[0]
#     Sigma_list[j]=result1[1]
#     beta_list[j]=result1[2]
#     delta_list[j]=result1[3]
#     theta_list[j]=result1[4]
T=10000
# mu_lists,Sigma_lists,beta_lists,delta_lists,theta_lists=testGPPE.gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T)
# print("finished")




params_list=[]
for j in range(len(mu_list)):
    params=testGPPE.gpp_estimation.argument2vector(mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
    params=params.reshape(-1,1)
    params_list.append(params)


with open('/home/shij0d/Documents/Dis_Spatial/test/global_minizer.pkl', 'rb') as f:
    global_minizer = pickle.load(f)
mu=global_minizer['mu']
Sigma=global_minizer['Sigma']
beta=global_minizer['beta']
delta=global_minizer['delta']
theta=global_minizer['theta']
params=testGPPE.gpp_estimation.argument2vector(mu,Sigma,beta,delta,theta)
params=params.reshape(-1,1)
# params_list=[]
# for j in range(10):
#     params=testGPPE.gpp_estimation.argument2vector(mu,Sigma,beta,delta,theta)
#     params=params.reshape(-1,1)
#     params_list.append(params)






params=testGPPE.gpp_estimation.argument2vector(mu_list[1],Sigma_list[1],beta_list[1],delta_list[1],theta_list[1])
J=len(mu_list)
for j in range(1,J):
    params+=testGPPE.gpp_estimation.argument2vector(mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
params=params.reshape(-1,1)
params=params/J
params[-1]=3
#params_list=testGPPE.gpp_estimation.dis_opimize_stage1(params,T=100)
params.squeeze_()
mu,Sigma,beta,delta,theta=testGPPE.gpp_estimation.vector2arguments(params)
mu_list = []
Sigma_list = []
beta_list = []
delta_list = []
theta_list = []
for j in range(10):
    mu_list.append(mu)
    Sigma_list.append(Sigma)
    beta_list.append(beta)
    delta_list.append(delta)
    theta_list.append(theta)

testGPPE.gpp_estimation.de_optimize_stage2(mu_list,Sigma_list,beta_list,delta_list,theta_list,T=100)
