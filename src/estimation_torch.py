from __future__ import annotations
import numpy as np
import random
from src.kernel import Kernel_with_Grad,Matern_with_Grad
from sklearn.gaussian_process.kernels import Matern
import math
import numpy as np
import random
from typing import Callable, Any, List
from src.utils import softplus_torch,inv_softplus_torch,replace_negative_eigenvalues_with_zero,softplus_d
from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian,jacobian
import torch.multiprocessing as mp
from joblib import Parallel, delayed
import time

torch.autograd.set_detect_anomaly(True)


class GPPEstimation:
    def __init__(self, dis_data: list[torch.Tensor|np.ndarray], kernel: callable, knots: torch.Tensor|np.ndarray, weights: torch.Tensor|np.ndarray) -> None:
        """
        Initialize the class with distributed data, a kernel function, knots, and weights.
        
        Parameters:
        dis_data (list[torch.tensor]): List of tensors, each representing local data.
        kernel (callable): Kernel function to be used.
        knots (torch.tensor): Tensor representing knot points.
        weights (torch.tensor): Tensor representing weights.
        """
        for i in range(len(dis_data)):
            if not isinstance(dis_data[i],torch.Tensor):
                dis_data[i]=torch.tensor(dis_data[i],dtype=torch.float64)

        if not isinstance(knots,torch.Tensor):
            knots=torch.tensor(knots,dtype=torch.float64)

        if not isinstance(weights,torch.Tensor):
            weights=torch.tensor(weights,dtype=torch.float64)

        self.dis_data = dis_data  # List of distributed data tensors
        self.J = len(dis_data)    # Number of distributed data points
        self.knots = knots        # Knot points tensor
        self.m = len(self.knots)  # Number of knots
        self.p = dis_data[0].shape[1] - 3  # Dimensionality of the data minus 3
        self.weights = weights    # Weights tensor
        self.kernel = kernel      # Kernel function
    
    def vector2arguments(self, params:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """
        Converts a parameter vector into individual arguments.
        
        Args:
            params (torch.Tensor): 1D array of parameters.
        
        Returns:
            tuple: A tuple containing (mu, Sigma, beta, delta, theta) where theta is (alpha, length_scale).
        """
      
        
        start = 0

        # Extracting mu from params
        mu = params[:self.m].unsqueeze(1)
        start += self.m

        # Constructing the lower triangular matrix L
        L_ol = torch.zeros((self.m, self.m), dtype=torch.float64)
        indices = torch.tril_indices(row=self.m, col=self.m, offset=0)
        L_ol[indices[0], indices[1]] = params[start:(start + self.m * (self.m + 1) // 2)]
        start += self.m * (self.m + 1) // 2

        # Setting the diagonal elements to be the softplus of their original values
        L = L_ol.clone()
        diag_indices = torch.arange(self.m)
        L[diag_indices, diag_indices] = softplus_torch(L[diag_indices, diag_indices])
        Sigma = L @ L.T

        # Extracting beta from params
        beta = params[start:(start + self.p)].unsqueeze(1)
        start += self.p

        # Extracting and transforming delta from params
        delta_ol = params[start]
        delta = softplus_torch(delta_ol)
        start += 1

        # Extracting and transforming theta values from params
        theta_ol = params[start:]
        theta = softplus_torch(theta_ol).unsqueeze(1)

        return (mu, Sigma, beta, delta, theta)
    def argument2vector(self,mu:torch.Tensor, Sigma:torch.Tensor, beta:torch.Tensor, delta:torch.Tensor, theta:torch.Tensor)->torch.Tensor:
        params=torch.empty(self.m+self.m*(self.m+1)//2+self.p+1+theta.shape[0],dtype=torch.float64)
        start=0
        params[:self.m]=mu.squeeze()
        start+=self.m
        L=torch.linalg.cholesky(Sigma)
        L_ol=L.clone()
        diag_indices = torch.arange(self.m)
        L_ol[diag_indices, diag_indices] = inv_softplus_torch(L_ol[diag_indices, diag_indices])
        tri_indices=torch.tril_indices(row=L_ol.shape[0], col=L_ol.shape[1], offset=0)
        params[start:(start + self.m * (self.m + 1) // 2)]=L_ol[tri_indices[0], tri_indices[1]]
        start += self.m * (self.m + 1) // 2

        params[start:(start+self.p)]=beta.squeeze()
        start+=self.p

        params[start]=inv_softplus_torch(delta)
        start += 1

        params[start:]=inv_softplus_torch(theta).squeeze()

        return params
    def vector2arguments_lik(self, params:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        start = 0

        # Extracting beta from params
        beta = params[start:start + self.p].unsqueeze(1)
        start += self.p

        # Extracting and transforming delta from params
        delta_ol=params[start]
        delta = softplus_torch(delta_ol)
        start += 1

        # Extracting and transforming theta values from params
        theta_ol = params[start:]
        theta = softplus_torch(theta_ol).unsqueeze(1)

        return (beta, delta, theta)
    def argument2vector_lik(self,beta:torch.Tensor, delta:torch.Tensor, theta:torch.Tensor)->torch.Tensor:
        
        params=torch.empty(self.p+1+theta.shape[0],dtype=torch.float64)
        
        start=0

        params[start:(start+self.p)]=beta.squeeze()
        start+=self.p

        params[start]=inv_softplus_torch(delta)
        start += 1

        params[start:]=inv_softplus_torch(theta).squeeze()
        return params
    def local_fun(self, local_locs:torch.Tensor, local_z:torch.Tensor, local_X:torch.Tensor, params:torch.Tensor|np.array, requires_grad=True):
        """
        Compute the local objective function value and optionally its gradient.

        local_locs: 2D array
        local_z: 1D array
        local_X: 2D array
        params: 1D array
        requires_grad: Boolean to indicate if gradient computation is needed
        """
        # Convert input data to torch tensors
      
        knots=self.knots
        n = local_z.shape[0]
        
        # Convert params to torch tensor with gradient tracking
        params.requires_grad_(requires_grad)
        if params.grad is not None:
            params.grad.data.zero_()
        # Extract parameters
        mu, Sigma, beta, delta, theta = self.vector2arguments(params)
        
        
        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        B = self.kernel(local_locs, knots, theta) @ torch.linalg.inv(K)
        
        # Compute the value of the local objective function
        errorV= local_X @ beta-local_z
        f_value = -n * torch.log(delta) + delta * (
            torch.trace(B.T @ B @ (Sigma + mu @ mu.T)) 
            + 2 * errorV.T @ B @ mu 
            + errorV.T @ errorV
        )
        
        
        if requires_grad:
            # Compute the gradients
            f_value.backward()
            grad = params.grad
            return f_value, grad
        else:
            return f_value
    def local_fun_wrapper(self,j:int,requires_grad=True):
        """
        Local objectives in each machine.
        
        Returns:
            List[Callable]: A list of local objective functions.
        """
        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
        if requires_grad:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.local_fun(local_locs,local_z,local_X,params,False)
                return value.numpy().flatten()
            def gradf(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                _,grad=self.local_fun(local_locs,local_z,local_X,params,True)
                return grad.numpy()
            return fun,gradf
        else:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.local_fun(local_locs,local_z,local_X,params,False)
                return value.numpy().flatten()
            return fun
    def local_value(self,j,mu:torch.Tensor,Sigma:torch.Tensor,beta:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor):
        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:] 
        knots=self.knots
        n = local_z.shape[0]
        
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        
        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        B = self.kernel(local_locs, knots, theta) @ torch.linalg.inv(K)
        
        # Compute the value of the local objective function
        errorV= local_X @ beta-local_z
        f_value = -n * torch.log(delta) + delta * (
            torch.trace(B.T @ B @ (Sigma + mu @ mu.T)) 
            + 2 * errorV.T @ B @ mu 
            + errorV.T @ errorV
        )        
        return  f_value 
    
    def local_grad_theta(self,j,mu:torch.Tensor,Sigma:torch.Tensor,beta:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor):
        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:] 
        knots=self.knots
        n = local_z.shape[0]
        
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        
        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        B = self.kernel(local_locs, knots, theta) @ torch.linalg.inv(K)
        
        # Compute the value of the local objective function
        errorV= local_X @ beta-local_z
        f_value = -n * torch.log(delta) + delta * (
            torch.trace(B.T @ B @ (Sigma + mu @ mu.T)) 
            + 2 * errorV.T @ B @ mu 
            + errorV.T @ errorV
        )        
        f_value.backward()
        grad = theta.grad
        return  grad 
    

    def local_hessian_theta(self,j,mu:torch.Tensor,Sigma:torch.Tensor,beta:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor):
        
        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:] 
        knots=self.knots
        n = local_z.shape[0]
        errorV= local_X @ beta-local_z
        def local_f(theta_v):
            # Compute the kernel matrices
            K = self.kernel(knots, knots, theta_v)
            B = self.kernel(local_locs, knots, theta_v) @ torch.linalg.inv(K)
            
            # Compute the value of the local objective function
            f_value = -n * torch.log(delta) + delta * (
                torch.trace(B.T @ B @ (Sigma + mu @ mu.T)) 
                + 2 * errorV.T @ B @ mu 
                + errorV.T @ errorV
            )   
            f_value=f_value.squeeze()  
            return f_value 
        
        Hessian=hessian(local_f,theta.squeeze())
        
        return Hessian
    
    def com_fun(self,params:torch.Tensor,requires_grad=True):
        """
        Define a common function known to all machines.

        Args:
            params (np.array): Parameters used to compute the function.
            requires_grad (bool): Whether to compute gradients. Default is True.

        Returns:
            float or tuple: The computed function value, optionally with gradients if requires_grad=True.
        """
        
        # Convert params to torch tensor with gradient tracking
        if params.grad is not None:
            params.grad.data.zero_()

        params.requires_grad_(requires_grad)
        
        mu, Sigma, _, _, theta = self.vector2arguments(params)

      
        # Compute the kernel matrices
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))

        if requires_grad:
            # Compute the gradients
            f_value.backward()
            grad = params.grad
            return f_value, grad
        else:
            return f_value
    
    def com_fun_wrapper(self,requires_grad=True) -> List[Callable]:
        """
        Wrapper function to generate callable functions for com_fun.

        Args:
            requires_grad (bool): Whether to include gradient functions. Default is True.

        Returns:
            List[Callable]: A list of callable functions.
        """
        if requires_grad:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.com_fun(params,False)
                return value.numpy().flatten()
            def gradf(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                _,grad=self.com_fun(params,True)
                return grad.numpy()
            return fun,gradf
        else:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.com_fun(params,False)
                return value.numpy().flatten()
            return fun
    
    def com_value(self,mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))

        return f_value
    
    def com_grad_theta(self,mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        #f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))
        f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) +torch.log( torch.det(K))
        f_value.backward()
        grad = theta.grad
        return grad
    
    def com_hessian_theta(self,mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):

        def com_f1(theta_v:torch.Tensor):
            K = self.kernel(self.knots, self.knots, theta_v)
            invK = torch.inverse(K)
            f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) 
            return f_value
        def det(theta_v):
            K = self.kernel(self.knots, self.knots, theta_v)
            det=torch.det(K)
            return det
        def hessian2(theta_v:torch.Tensor):
            value=det(theta_v)
            jac=jacobian(det,theta)
            hes=hessian(det,theta_v)
            jac_scaled=jac/value
            jac_scaled=jac_scaled.reshape(-1,1)
            hes2=hes/value-jac_scaled@jac_scaled.T
            return hes2
            
        hes1=hessian(com_f1,theta.squeeze())
        hes2=hessian2(theta.squeeze())
        Hessian=hes1+hes2
        return Hessian

    def neg_log_lik(self, local_locs:torch.Tensor, local_z:torch.Tensor, local_X:torch.Tensor, params:torch.Tensor, requires_grad=True):
      
        knots=self.knots
        n = local_z.shape[0]
        
        # Convert params to torch tensor with gradient tracking
        params.requires_grad_(requires_grad)
        if params.grad is not None:
            params.grad.data.zero_()
        
        beta, delta, theta = self.vector2arguments_lik(params)
       

        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        invK=torch.linalg.inv(K)
        B = self.kernel(local_locs, knots, theta) @ invK

        

        tempM=invK+delta*B.T@B
        errorv=local_X@beta-local_z

        f_value=torch.log(torch.linalg.det(tempM))+torch.log(torch.linalg.det(K))-n*torch.log(delta)+delta*(errorv.T)@errorv-delta**2*(errorv.T@B@torch.linalg.inv(tempM)@B.T@errorv)
        f_value=f_value/n

        if requires_grad:
            # Compute the gradients
            f_value.backward()
            grad = params.grad
            return f_value, grad
        else:
            return f_value

    def local_neg_log_lik_wrapper(self,j,requires_grad=True):
        """
        the negative local log likelihood function for the low rank model 
        """

        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:]
        if requires_grad:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.neg_log_lik(local_locs,local_z,local_X,params,False)
                return value.numpy().flatten()
            def gradf(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                _,grad=self.neg_log_lik(local_locs,local_z,local_X,params,True)
                return grad.numpy()
                
            return fun,gradf
        else:
            
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.neg_log_lik(local_locs,local_z,local_X,params,False)
                return value.numpy().flatten()
            return fun
    def neg_log_lik_wrapper(self,requires_grad=True):
        """
        the negative local log likelihood function for the low rank model 
        """
        local_locs_list = []
        local_z_list=[]
        local_X_list=[]
        N=0
        for data in self.dis_data:
            local_locs = data[:, :2]  # Extract the first two columns as local locations
            local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
            local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
            n = len(local_z)  
            local_locs_list.append(local_locs)
            local_z_list.append(local_z)
            local_X_list.append(local_X)
            N+=n   
        locs = torch.cat(local_locs_list, dim=0)    
        z = torch.cat(local_z_list, dim=0)
        X = torch.cat(local_X_list, dim=0)

        if requires_grad:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.neg_log_lik(locs,z,X,params,False)
                return value.numpy().flatten()   
            def gradf(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                _,grad=self.neg_log_lik(locs,z,X,params,True)
                return grad.numpy()

            return fun,gradf
        else:
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.neg_log_lik(locs,z,X,params,False)
                return value.numpy().flatten()
            return fun
    
    def get_local_pos(self,j:int,params_lik:torch.Tensor):
        #list of 1D params
    
        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].reshape(-1,1)      # Extract the third column as local z
        local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
        n = len(local_z)  
        beta,delta,theta=self.vector2arguments_lik(params_lik)
        # Compute the kernel matrices
        K = self.kernel(self.knots, self.knots, theta)
        invK=torch.linalg.inv(K)
        B = self.kernel(local_locs, self.knots, theta) @ invK
        tempM=invK+delta*B.T@B
        errorv=local_X@beta-local_z
        Sigma=torch.linalg.inv(tempM/n)/n
        mu=Sigma@(delta*B.T@(-errorv))
     
        return (mu,Sigma)
        
    def get_pos(self,params_lik:torch.Tensor):

        beta,delta,theta=self.vector2arguments_lik(params_lik)
        local_locs_list = []
        local_z_list=[]
        local_X_list=[]
        N=0
       
        # Iterate through the distributed data
        for data in self.dis_data:
            local_locs = data[:, :2]  # Extract the first two columns as local locations
            local_z = data[:, 2].reshape(-1,1)       # Extract the third column as local z
            local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
            n = len(local_z)  
            local_locs_list.append(local_locs)
            local_z_list.append(local_z)
            local_X_list.append(local_X)
            N+=n  
         
        locs = torch.cat(local_locs_list, dim=0)    
        z = torch.cat(local_z_list, dim=0)
        X = torch.cat(local_X_list, dim=0)
        K = self.kernel(self.knots, self.knots, theta)
        invK=torch.linalg.inv(K)
        B = self.kernel(locs, self.knots, theta) @ invK
        tempM=invK+delta*B.T@B
        errorv=X@beta-z
        Sigma=torch.linalg.inv(tempM/N)/N
        mu=Sigma@(delta*B.T@(-errorv))
       
        return (mu,Sigma)
    
    def get_local_minimizer(self,j:int,x0:torch.Tensor,thread_num=None):
        """
        Optimize the local likelihood functions in each machine to obtain the initial points
        """
        if thread_num!=None:
            torch.set_num_threads(thread_num) 
        x0=x0.numpy()
        loc_nllikf,loc_nllikgf=self.local_neg_log_lik_wrapper(j,requires_grad=True) 
        options = {'maxiter': 100} 
        result=minimize(fun=loc_nllikf,
                     x0=x0,
                     method="CG",
                     jac=loc_nllikgf,options=options)
        x0=result.x
        result=minimize(fun=loc_nllikf,
                     x0=x0,
                     method="BFGS",
                     jac=loc_nllikgf)
        local_minimizer_lik=torch.tensor(result.x,dtype=torch.float64)
        mu,Sigma=self.get_local_pos(j,local_minimizer_lik)
        beta,delta,theta=self.vector2arguments_lik(local_minimizer_lik)
        #local_minimizer=self.argument2vector(mu,Sigma,beta,delta,theta)

        return (mu,Sigma,beta, delta, theta,result)
    def get_local_minimizers_parallel(self, x0,job_num,thread_num=None):
        mu_list = []
        Sigma_list = []
        beta_list = []
        delta_list = []
        theta_list = []
        result_list = []
        except_list = []

        def compute_minimizer(j):
            try:
                mu, Sigma, beta, delta, theta, result = self.get_local_minimizer(j, x0,thread_num)
                return mu, Sigma, beta, delta, theta, result, None  # No exception
            except Exception as e:
                return None, None, None, None, None, None, j  # Capture the exception
        

        # Parallelize over range(self.J) using joblib
        results = Parallel(n_jobs=job_num)(delayed(compute_minimizer)(j) for j in range(self.J))

        

        # Process results
        for mu, Sigma, beta, delta, theta, result, exception in results:
            if exception is None:
                mu_list.append(mu)
                Sigma_list.append(Sigma)
                beta_list.append(beta)
                delta_list.append(delta)
                theta_list.append(theta)
                result_list.append(result)
            else:
                except_list.append(exception)

        return mu_list, Sigma_list, beta_list, delta_list, theta_list, result_list, except_list
    def get_local_minimizers(self,x0:torch.Tensor):
        mu_list=[]
        Sigma_list=[]
        beta_list=[]
        delta_list=[]
        theta_list=[]
        result_list=[]
        except_list=[]
        for j in range(self.J):
            try:
                mu,Sigma,beta, delta, theta,result=self.get_local_minimizer(j,x0)
                mu_list.append(mu)
                Sigma_list.append(Sigma)
                beta_list.append(beta)
                theta_list.append(theta)
                delta_list.append(delta)
                result_list.append(result)
            except Exception:
                except_list.append(j)
        return mu_list,Sigma_list,beta_list,delta_list,theta_list,result_list,except_list
    def get_minimier(self,x0:torch.Tensor,thread_num=None):
        if thread_num!=None:
            torch.set_num_threads(thread_num) 
        x0=x0.numpy()
        nllikf,nllikgf=self.neg_log_lik_wrapper(requires_grad=True)  
        result=minimize(fun=nllikf,
                     x0=x0,
                     method="BFGS",
                     jac=nllikgf,tol=1e-10)
        minimizer_lik=torch.tensor(result.x,dtype=torch.float64)
        mu,Sigma=self.get_pos(minimizer_lik)
        beta, delta, theta=self.vector2arguments_lik(minimizer_lik)
        
        return (mu,Sigma,beta, delta, theta,result)
    

    def dis_opimize_stage1(self,initial_params:torch.Tensor,T:int,noisy=True):
        step_size=0.1
        epsilon=0.001
        history_size=10
        length=initial_params.shape[0]
        initial_params.squeeze_()
        def sum_grad(params:torch.Tensor):
            grad=torch.zeros((length,1),dtype=torch.double)
            params=np.squeeze(params.numpy())
            for j in range(self.J):
                _,gradf=self.local_fun_wrapper(j)
                local_gad=gradf(params).reshape(-1,1)
                grad+=torch.tensor(local_gad,dtype=torch.double)
            return grad
        
        def com_grad(params:torch.Tensor):
            params=np.squeeze(params.numpy())
            _,gradf=self.com_fun_wrapper()
            return gradf(params).reshape(-1,1)
        
        d = initial_params.numel()
        s_list = []  # List to store s_t
        y_bar_list = []  # List to store y_bar_t
        rho_list = []  # List to store rho_t
        alpha_list = []  # List to store alpha_t
        q = torch.zeros(d,dtype=torch.double)
       
        
        params=initial_params
        params_list=[params]
        
        for t in range(T):
            _,_,_,_,theta=self.vector2arguments(params.squeeze())
            
            
            step_size_g=torch.ones(length)*100
            # step_size_g[-2]=1000
            # if t<=200:
            #     step_size_g[-1]=2
            # else:
            #     step_size_g[-1]=200
            step_size_qn=1
            step_size_bb=0.1
            if noisy:
                noise=torch.randn(size=(length,))*epsilon/math.sqrt(length)
            else:
                noise=0.0
            
                #y_local_grad=weights_kron_f(length)@(y_local_grad+local_grads(params_Mstack_list[t])-local_grads(params_Mstack_list[t-1]))
            if t==0:
                grad=(sum_grad(params)+com_grad(params))/10**4
                grad.squeeze_()
                grad_noise=grad+noise
                p_t=-grad_noise
                bb=1
            else:
                
                q[:]=grad_noise
                for i in range(len(s_list) - 1, -1, -1):
                    alpha_i = rho_list[i] * torch.dot(s_list[i], q)
                    alpha_list.append(alpha_i)
                    q -= alpha_i * y_bar_list[i]
                r=q.clone()
                
                for i in range(len(s_list)):
                    beta = rho_list[i] * torch.dot(y_bar_list[i], r)
                    r += s_list[i] * (alpha_list[i] - beta)
                p_t = -r
                alpha_list.clear()
                
                #bb
                bb=s_list[-1]/y_bar_list[-1]
                bb=bb.abs()

            params_g=params-step_size_g*grad_noise
            #parms_qn=params+step_size_qn*p_t
            params_bb=params-step_size_bb*bb*grad_noise
            
            grad_g=(sum_grad(params_g)+com_grad(params_g))/10**4
            # _,_,_,_,theta_qn=self.vector2arguments(parms_qn.squeeze())
            # exception_occurred = False  # Initialize the boolean variable

            # try:
            #     grad_qn = (sum_grad(parms_qn) + com_grad(parms_qn)) / 10**4
            # except NameError as e:
            #     print(f"NameError: {e}")
            #     exception_occurred = True
            # except TypeError as e:
            #     print(f"TypeError: {e}")
            #     exception_occurred = True
            # except AttributeError as e:
            #     print(f"AttributeError: {e}")
            #     exception_occurred = True
            # except ZeroDivisionError as e:
            #     print(f"ZeroDivisionError: {e}")
            #     exception_occurred = True
            # except Exception as e:
            #     print(f"An unexpected error occurred: {e}")
            #     exception_occurred = True
                
            exception_occurred_bb = False  # Initialize the boolean variable
            try:
                grad_bb = (sum_grad(params_bb) + com_grad(params_bb)) / 10**4
            except Exception as e:
                #print(f"An unexpected error occurred: {e}")
                exception_occurred_bb = True
                
            grad_norm_g=torch.norm(grad_g)
            grad_norm_bb=torch.norm(grad_bb)
            
            
            if exception_occurred_bb==True or grad_norm_g<grad_norm_bb:
                params_new=params_g
                grad_new=grad_g
                #print("gradient descent method")
            else:
                params_new=params_bb
                grad_new=grad_bb
                #print("BB")
            # params_new=params_g
            # grad_new=grad_g
            # grad_norm_g=torch.norm(grad_g)
            # grad_norm_qn=torch.norm(grad_qn)
            # if exception_occurred==False or grad_norm_g<grad_norm_qn:
            #     params_new=params_g
            #     grad_new=grad_g
            #     print("gradient descent method")
            # else:
            #     params_new=parms_qn
            #     grad_new=grad_qn
            #     print("LBFGS")

            # params_new=params+step_size*p_t
            # grad_new=(sum_grad(params_new)+com_grad(params_new))/10**4
            grad_new.squeeze_()
            s_t=params_new-params
            y_t=grad_new-grad
            
            gamma_t0=max(torch.dot(y_t, y_t) / torch.dot(y_t, s_t),1)
            #print("gamma_t0:",torch.dot(y_t, y_t) / torch.dot(y_t, s_t))
            h_t0=1/gamma_t0

            temp1=torch.dot(y_t, s_t)
            temp2=torch.dot(s_t, s_t)/h_t0

            if temp1<0.25* temp2:
                zeta_t=0.75* temp2/(temp2-temp1)
            else:
                zeta_t=1
            zeta_t=1    
            #print("zeta_t:",zeta_t)
            y_bar_t=zeta_t*y_t+(1-zeta_t)*s_t/h_t0   
            
            rho_t = 1.0 / torch.dot(y_bar_t, s_t)
            
            if len(s_list) == history_size:
                s_list.pop(0)
                y_bar_list.pop(0)
                rho_list.pop(0)
            s_list.append(s_t)
            y_bar_list.append(y_bar_t)
            rho_list.append(rho_t)
            
            
            grad_new_norm=torch.norm(grad_new)
            grad_norm=torch.norm(grad)
            # if grad_new_norm < 1e-5:
            #     break
            #print(f'Iteration {t}, theta = {theta}, grad_norm = {grad_norm}')

            # if grad_new_norm<grad_norm*0.5:
            #     step_size=step_size*1.5
            # elif grad_new_norm>grad_norm:
            #     step_size=step_size*0.5
            # print(step_size)
            params=params_new
            grad=grad_new
            grad_noise=grad+noise
            #params=params-torch.mul(grad_noise,step_size)
            params_list.append(params)
            if t%10==0:
                #print("theta,",theta.numpy())
                print(f'Iteration {t}, theta = {theta.numpy()}, grad_norm = {grad_norm}')
            
        return params_list
    
    def de_opimize_stage1(self,initial_params_list:list[torch.Tensor],T:int):
        step_size=0.2
        epsilon=0.01
        length=initial_params_list[0].shape[0]
        def local_grads(params_Mstack:torch.Tensor):
            grad_Mstack=torch.zeros((self.J*length,1),dtype=torch.double)
            for j in range(self.J):
                _,gradf=self.local_fun_wrapper(j)
                params=np.squeeze(params_Mstack[(length*j):(length*(j+1)),:].numpy())
                local_gad=gradf(params).reshape(-1,1)
                grad_Mstack[(length*j):(length*(j+1)),:]=torch.tensor(local_gad,dtype=torch.double)
            return grad_Mstack
        
        def com_grads(params_Mstack:torch.Tensor):
            grad_Mstack=torch.zeros((self.J*length,1),dtype=torch.double)
            _,gradf=self.com_fun_wrapper()
            for j in range(self.J):
                params=np.squeeze(params_Mstack[(length*j):(length*(j+1)),:].numpy())
                local_gad=gradf(params).reshape(-1,1)
                grad_Mstack[(length*j):(length*(j+1)),:]=torch.tensor(local_gad,dtype=torch.double)
            return grad_Mstack
        
        def weights_kron_f(num,iter=1):
            weight=self.weights
            for i in range(iter):
                weight=weight@weight
            weights_kron=torch.kron(weight,torch.eye(num,dtype=torch.double))
            return weights_kron
        
        params_Mstack=torch.zeros((self.J*length,1),dtype=torch.double)
        for j in range(self.J):
            params_Mstack[(length*j):(length*(j+1)),:]=initial_params_list[j]
        
        params_Mstack=weights_kron_f(length)@params_Mstack
        params_Mstack_list=[params_Mstack]
        
        for t in range(T):
            _,_,_,_,theta=self.vector2arguments(params_Mstack[0:length].squeeze())
            
            print("theta,",theta.numpy())
            step_size=torch.ones((length,1))*2
            step_size[-2]=100
            step_size[-1]=3
            noise=torch.randn(size=(length*self.J,1))*epsilon/length
            if t==0:
                y_local_grad=weights_kron_f(length)@local_grads(params_Mstack_list[t])
            else:
                y_local_grad_temp=torch.zeros((length*self.J,1),dtype=torch.double)
                y_t=local_grads(params_Mstack_list[t])
                y_t_1=local_grads(params_Mstack_list[t-1])
                for j in range(self.J):
                    tempv=torch.zeros((length,1),dtype=torch.double)
                    for i in range(self.J):
                        tempv+=self.weights[j,i]*(y_local_grad[(length*i):(length*(1+i)),:]+y_t[(length*i):(length*(1+i)),:]-y_t_1[(length*i):(length*(1+i)),:])
                    y_local_grad_temp[(length*j):(length*(j+1)),:]=tempv
                y_local_grad=y_local_grad_temp
                #y_local_grad=weights_kron_f(length)@(y_local_grad+local_grads(params_Mstack_list[t])-local_grads(params_Mstack_list[t-1]))
            y_grad=(y_local_grad*self.J+com_grads(params_Mstack_list[t]))/10**4
            y_grad_noise=y_grad#+noise
            params_Mstack_temp=torch.zeros((length*self.J,1),dtype=torch.double)
            for j in range(self.J):
                tempv=torch.zeros((length,1),dtype=torch.double)
                for i in range(self.J):
                    tempv+=self.weights[j,i]*(params_Mstack[(length*i):(length*(1+i)),:]-torch.mul(y_grad_noise[(length*i):(length*(1+i)),:],step_size))
                params_Mstack_temp[(length*j):(length*(j+1)),:]=tempv
            params_Mstack=params_Mstack_temp
            #params_Mstack=weights_kron_f(length)@(params_Mstack-step_size*y_grad_noise)
            params_Mstack_list.append(params_Mstack)
            print(y_grad[5156:5158,0].numpy())
            
            

        return params_Mstack_list
    
    def de_optimize_stage2(self,mu_list:list[torch.Tensor],Sigma_list:list[torch.Tensor],beta_list:list[torch.Tensor],delta_list:list[torch.Tensor],theta_list:list[torch.Tensor],T:int,weights_round=4,seed=2024):
        torch.manual_seed(seed)
        #self.weights=torch.ones((self.J,self.J),dtype=torch.double)/self.J
        self.weights=torch.matrix_power(self.weights, weights_round)
        #define some functions
        def K_f(theta):
            return self.kernel(self.knots, self.knots, theta)
        
        def local_B_f(j,theta):
            data=self.dis_data[j]
            local_locs = data[:, :2]
            K = K_f(theta)
            invK=torch.linalg.inv(K)
            B = self.kernel(local_locs, self.knots, theta) @ invK
            return B
        
        def local_size_f(j):
            return self.dis_data[j].shape[0]
        
        def local_erorrV_f(j,beta):
            data=self.dis_data[j]
            local_z = data[:, 2].reshape(-1,1)      # Extract the third column as local z
            local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
            errorv=local_X@beta-local_z
            return errorv
        
        def local_X_f(j):
            data=self.dis_data[j]
            local_X = data[:, 3:]
            return local_X
        
        def local_z_f(j):
            data=self.dis_data[j]
            local_z = data[:, 2].reshape(-1,1)
            return local_z
        
        
        y_XTX_Mstack = torch.zeros((self.p, self.p, self.J), dtype=torch.double)
        def compute_XTX_j(j):
            local_X = local_X_f(j)
            return local_X.T @ local_X
        # Parallel execution for each j
        results = Parallel(n_jobs=-1)(delayed(compute_XTX_j)(j) for j in range(self.J))
        # Stack the results
        for j in range(self.J):
            y_XTX_Mstack[:, :, j] = results[j]
        
        # y_XTX_Mstack=torch.zeros((self.p,self.p,self.J),dtype=torch.double)
        # for j in range(self.J):
        #     local_X=local_X_f(j)
        #     y_XTX_Mstack[:,:,j]=local_X.T@local_X

            
        def y_mu_f_parallel(beta_list, theta_list):
            y_mu_Mstack = torch.zeros((self.m, self.J), dtype=torch.double)
            
            def compute_j(j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                return -local_B.T @ local_errorV

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(j, beta_list[j], theta_list[j]) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_mu_Mstack[:, j:(j+1)] = results[j]
            
            return y_mu_Mstack
        
        def y_mu_f(beta_list,theta_list):
            y_mu_Mstack=torch.zeros((self.m,self.J),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                local_erorrV=local_erorrV_f(j,beta_list[j])
                y_mu_Mstack[:,j:(j+1)]=-local_B.T@local_erorrV
            return y_mu_Mstack
        
        def y_Sigma_f_parallel(theta_list):
            y_Sigma_Mstack = torch.zeros((self.m, self.m, self.J), dtype=torch.double)
            
            def compute_j(j, theta_j):
                local_B = local_B_f(j, theta_j)
                return local_B.T @ local_B

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(j, theta_list[j]) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_Sigma_Mstack[:, :, j] = results[j]
            
            return y_Sigma_Mstack

        def y_Sigma_f(theta_list):
            y_Sigma_Mstack=torch.zeros((self.m,self.m,self.J),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                y_Sigma_Mstack[:,:,j]=local_B.T@local_B
            return y_Sigma_Mstack
        
        def y_beta_f_parallel(mu_list, theta_list):
            y_beta_Mstack = torch.zeros((self.p, self.J), dtype=torch.double)
            
            def compute_j(j, mu_j, theta_j):
                local_X = local_X_f(j)
                local_B = local_B_f(j, theta_j)
                local_z = local_z_f(j)
                return local_X.T @ (local_z - local_B @ mu_j)

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(j, mu_list[j], theta_list[j]) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_beta_Mstack[:, j:(j+1)] = results[j]
            
            return y_beta_Mstack
        
        def y_beta_f(mu_list,theta_list):
            y_beta_Mstack=torch.zeros((self.p,self.J),dtype=torch.double)
            for j in range(self.J):
                local_X=local_X_f(j)
                local_B=local_B_f(j,theta_list[j])
                local_z=local_z_f(j)
                y_beta_Mstack[:,j:(j+1)]=local_X.T@(local_z-local_B@mu_list[j])
            return y_beta_Mstack
        
        def y_delta_f_parallel(mu_list, Sigma_list, beta_list, theta_list):
            y_delta_Mstack = torch.zeros((1, self.J), dtype=torch.double)
            
            def compute_j(j, mu_j, Sigma_j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                
                term1 = torch.trace(local_B.T @ local_B @ (Sigma_j + mu_j @ mu_j.T))
                term2 = 2 * local_errorV.T @ local_B @ mu_j
                term3 = local_errorV.T @ local_errorV
                
                return term1 + term2 + term3

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(j, mu_list[j], Sigma_list[j], beta_list[j], theta_list[j]) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_delta_Mstack[:, j] = results[j]
            
            return y_delta_Mstack

        def y_delta_f(mu_list,Sigma_list,beta_list,theta_list):
            y_delta_Mstack=torch.zeros((1,self.J),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                local_erorrV=local_erorrV_f(j,beta_list[j])
                y_delta_Mstack[:,j]=torch.trace(local_B.T@local_B@(Sigma_list[j]+mu_list[j]@mu_list[j].T))+2*local_erorrV.T@local_B@mu_list[j]+local_erorrV.T@local_erorrV
            return y_delta_Mstack
        
        # def y_value_f(mu_list,Sigma_list,beta_list,delta_list,theta_list):
        #     y_value_M=torch.zeros((1,1),dtype=torch.double)
        #     for j in range(self.J):
        #         local_value=self.local_value(j,mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
        #         y_value_M+=local_value.reshape(-1,1)
        #     y_value_M=y_value_M/self.J
        #     return y_value_M
        # def com_value_f(mu_list,Sigma_list,theta_list):
        #     com_value_M=torch.zeros((1,1),dtype=torch.double)
        #     for j in range(self.J):
        #         com_value=self.com_value(mu_list[j],Sigma_list[j],theta_list[j])            
        #         com_value_M+=com_value.reshape(-1,1)
        #     com_value_M=com_value_M/self.J
        #     return com_value_M
        
        

        def y_theta_f_parallel(mu_list, Sigma_list, beta_list, delta_list, theta_list):
          
            y_theta_Mstack = torch.zeros((2, self.J), dtype=torch.double)

            def compute_local_grad(j, mu, Sigma, beta, delta, theta):
                local_g_theta = self.local_grad_theta(j, mu, Sigma, beta, delta, theta)
                return j, local_g_theta.reshape(-1, 1)

            # Parallelize the computation using joblib
            results = Parallel(n_jobs=-1)(delayed(compute_local_grad)(j, mu_list[j], Sigma_list[j], beta_list[j], delta_list[j], theta_list[j]) for j in range(self.J))

            # Populate y_theta_Mstack with the results
            for j, local_g_theta in results:
                y_theta_Mstack[:, j:(j + 1)] = local_g_theta

            # Optionally move results back to CPU
            return y_theta_Mstack  # Return as a CPU tensor if needed
  
        def y_theta_f(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            y_theta_Mstack=torch.zeros((2,self.J),dtype=torch.double)
            for j in range(self.J):
                local_g_theta=self.local_grad_theta(j,mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
                y_theta_Mstack[:,j:(j+1)]=local_g_theta.reshape(-1,1)
            return y_theta_Mstack
        
        def y_hessian_theta_f_parallel(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            hessian_theta_Mstack=torch.zeros((2,2,self.J),dtype=torch.double)
            def compute_local_hessian(j, mu, Sigma, beta, delta, theta):
                local_h_theta = self.local_hessian_theta(j, mu, Sigma, beta, delta, theta)
                return j, local_h_theta
            results = Parallel(n_jobs=-1)(delayed(compute_local_hessian)(j, mu_list[j], Sigma_list[j], beta_list[j], delta_list[j], theta_list[j]) for j in range(self.J))

            for j, local_h_theta in results:
                hessian_theta_Mstack[:,:,j]=local_h_theta

            return hessian_theta_Mstack
        
        
        def y_hessian_theta_f(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            hessian_theta_Mstack=torch.zeros((2,2,self.J),dtype=torch.double)
            for j in range(self.J):
                local_h_theta=self.local_hessian_theta(j,mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
                hessian_theta_Mstack[:,:,j]=local_h_theta
            return hessian_theta_Mstack
        
        def com_grad_theta_f_parallel(mu_list,Sigma_list,theta_list):
            com_grad_theta_Mstack=torch.zeros((2,self.J),dtype=torch.double)
            def compute_local_grad(j, mu, Sigma, theta):
                local_g_theta = self.com_grad_theta(mu, Sigma, theta)
                return j, local_g_theta.reshape(-1, 1)
            results = Parallel(n_jobs=-1)(delayed(compute_local_grad)(j, mu_list[j], Sigma_list[j], theta_list[j]) for j in range(self.J))

            for j, local_g_theta in results:
                com_grad_theta_Mstack[:, j:(j + 1)] = local_g_theta

            return com_grad_theta_Mstack
        

        def com_grad_theta_f(mu_list,Sigma_list,theta_list):
            com_grad_theta_Mstack=torch.zeros((2,self.J),dtype=torch.double)
            for j in range(self.J):
                com_g_theta=self.com_grad_theta(mu_list[j],Sigma_list[j],theta_list[j])            
                com_grad_theta_Mstack[:,j:(j+1)]=com_g_theta.reshape(-1,1)
            return com_grad_theta_Mstack
        
        def com_hessian_theta_f_parallel(mu_list,Sigma_list,theta_list):
            com_hessian_theta_Mstack=torch.zeros((2,2,self.J),dtype=torch.double)
            def compute_local_hessian(j, mu, Sigma, theta):
                com_h_theta = self.com_hessian_theta(mu, Sigma, theta)
                return j, com_h_theta
            results = Parallel(n_jobs=-1)(delayed(compute_local_hessian)(j, mu_list[j], Sigma_list[j], theta_list[j]) for j in range(self.J))

            for j, com_h_theta in results:
                com_hessian_theta_Mstack[:,:,j]=com_h_theta
            return com_hessian_theta_Mstack

        def com_hessian_theta_f(mu_list,Sigma_list,theta_list):
            com_hessian_theta_Mstack=torch.zeros((2,2,self.J),dtype=torch.double)
            for j in range(self.J):
                com_h_theta=self.com_hessian_theta(mu_list[j],Sigma_list[j],theta_list[j])            
                com_hessian_theta_Mstack[:,:,j]=com_h_theta
            return com_hessian_theta_Mstack
        
       
        
        def list2Mstack(lt:list[torch.Tensor]):
            J=len(lt)
            
            if lt[0].dim()==0:
                Mstack=torch.zeros((1,J),dtype=torch.double)
                for j in range(J):
                    Mstack[:,j]=lt[j]
            elif lt[0].dim()==1:
                r=lt[0].shape[0]
                Mstack=torch.zeros((r,J),dtype=torch.double)
                for j in range(J):
                    Mstack[:,j]=lt[j]
            else:
                r=lt[0].shape[0]
                c=lt[0].shape[1]
                if c==1:
                    Mstack=torch.zeros((r,J),dtype=torch.double)
                    for j in range(J):
                        Mstack[:,j]=lt[j].squeeze()
                else:
                    Mstack=torch.zeros((r,c,J),dtype=torch.double)
                    for j in range(J):
                        Mstack[:,:,j]=lt[j]
            
            return Mstack
        
        def Mstack2list(Mstack:torch.Tensor):
            lt=[]
            J=Mstack.shape[-1]
            d=Mstack.dim()
            for j in range(J):
                if d==2:
                    if Mstack.shape[0]==1:
                        lt.append(Mstack[:,j])
                    else:
                        lt.append(Mstack[:,j].unsqueeze(1))
                else:
                    lt.append(Mstack[:,:,j])
            return lt
        
        def avg_local_minimizer(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            mu_Mstack=list2Mstack(mu_list)
            Sigma_Mstack=list2Mstack(Sigma_list)
            beta_Mstack=list2Mstack(beta_list)
            delta_Mstack=list2Mstack(delta_list)
            theta_Mstack=list2Mstack(theta_list)
            mu_Mstack=torch.tensordot(mu_Mstack, self.weights, dims=1)
            Sigma_Mstack=torch.tensordot(Sigma_Mstack, self.weights, dims=1)
            beta_Mstack=torch.tensordot(beta_Mstack, self.weights, dims=1)
            delta_Mstack=torch.tensordot(delta_Mstack, self.weights, dims=1)
            theta_Mstack=torch.tensordot(theta_Mstack, self.weights, dims=1)

            mu_list=Mstack2list(mu_Mstack)
            Sigma_list=Mstack2list(Sigma_Mstack)
            beta_list=Mstack2list(beta_Mstack)
            delta_list=Mstack2list(delta_Mstack)
            theta_list=Mstack2list(theta_Mstack)
            return mu_list,Sigma_list,beta_list,delta_list,theta_list
            

        size_stack=torch.zeros((1,self.J),dtype=torch.double) 
        for j in range(self.J):
            size_stack[:,j]=local_size_f(j)

        mu_list,Sigma_list,beta_list,delta_list,theta_list=avg_local_minimizer(mu_list,Sigma_list,beta_list,delta_list,theta_list)


        # mu_lists=[mu_list]
        # Sigma_lists=[Sigma_list]
        beta_lists=[beta_list]
        delta_lists=[delta_list]
        theta_lists=[theta_list]
        
        for t in range(T):
            mu_list_p=mu_list
            Sigma_list_p=Sigma_list
            if t%10==0:
                print(f"iteration:{t}", end=', ')
            #print(f"iteration:{t}")
            #mu and Sigma
            if t==0:
                y_mu_Mstack=torch.tensordot(y_mu_f_parallel(beta_lists[0],theta_lists[0]), self.weights, dims=1)
                y_Sigma_Mstack=torch.tensordot(y_Sigma_f_parallel(theta_lists[0]), self.weights, dims=1)
            else: 
                y_mu_Mstack=torch.tensordot(y_mu_Mstack+y_mu_f_parallel(beta_lists[t],theta_lists[t])-y_mu_f_parallel(beta_lists[t-1],theta_lists[t-1]), self.weights, dims=1)
                y_Sigma_Mstack=torch.tensordot(y_Sigma_Mstack+y_Sigma_f_parallel(theta_lists[t])-y_Sigma_f_parallel(theta_lists[t-1]), self.weights, dims=1)
            

            def compute_mu_sigma(j, theta_j, delta_j):
                y_Sigma = y_Sigma_Mstack[:, :, j]
                y_Sigma = replace_negative_eigenvalues_with_zero(y_Sigma)  # Ensure positive definiteness
                y_mu = y_mu_Mstack[:, j].unsqueeze(1)
                
                # Compute K and its inverse
                K = K_f(theta_j)
                invK = torch.linalg.inv(K)
                
                # Compute Sigma and mu
                Sigma = torch.linalg.inv(delta_j * self.J * y_Sigma + invK)
                mu = torch.linalg.inv(delta_j * y_Sigma + invK / self.J) @ (delta_j * y_mu)
                
                return mu, Sigma

            # Parallel execution for all j
            results = Parallel(n_jobs=-1)(delayed(compute_mu_sigma)(j, theta_lists[t][j], delta_lists[t][j]) for j in range(self.J))

            # Unpack the results
            mu_list, Sigma_list = zip(*results)
            mu_list = list(mu_list)
            Sigma_list = list(Sigma_list)
            # mu_lists.append(mu_list)
            # Sigma_lists.append(Sigma_list)


            # mu_list=[]
            # Sigma_list=[]
            # for j in range(self.J):
            #     y_Sigma=y_Sigma_Mstack[:,:,j]
            #     y_Sigma=replace_negative_eigenvalues_with_zero(y_Sigma) #make sure it is positive definite
            #     y_mu=y_mu_Mstack[:,j].unsqueeze(1)
            #     K=K_f(theta_lists[t][j])
            #     invK=torch.linalg.inv(K)
            #     Sigma=torch.linalg.inv(delta_lists[t][j]*self.J*y_Sigma+invK)
            #     mu=torch.linalg.inv(delta_lists[t][j]*y_Sigma+invK/self.J)*delta_lists[t][j]@y_mu
            #     mu_list.append(mu)
            #     Sigma_list.append(Sigma)
            # mu_lists.append(mu_list)
            # Sigma_lists.append(Sigma_list)



            y_XTX_Mstack=torch.tensordot(y_XTX_Mstack, self.weights, dims=1)
            if t==0:
                y_beta_Mstack=torch.tensordot(y_beta_f_parallel(mu_list,theta_lists[0]), self.weights, dims=1)
            else:
                y_beta_Mstack=torch.tensordot(y_beta_Mstack+y_beta_f_parallel(mu_list,theta_lists[t])-y_beta_f_parallel(mu_list_p,theta_lists[t-1]), self.weights, dims=1)
            beta_list=[]
            for j in range(self.J):
                y_XTX=y_XTX_Mstack[:,:,j]
                y_beta=y_beta_Mstack[:,j].unsqueeze(1)
                beta=torch.linalg.inv(y_XTX)@y_beta
                beta_list.append(beta)
            beta_lists.append(beta_list)


            #delta
            if t==0:
                y_delta_Mstack=torch.tensordot(y_delta_f_parallel(mu_list,Sigma_list,beta_lists[1],theta_lists[0]), self.weights, dims=1)
            else:
                y_delta_Mstack=torch.tensordot(y_delta_Mstack+y_delta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],theta_lists[t])-y_delta_f_parallel(mu_list_p,Sigma_list_p,beta_lists[t],theta_lists[t-1]), self.weights, dims=1)
          
            size_stack=torch.tensordot(size_stack, self.weights, dims=1)

            delta_Mstack=size_stack/y_delta_Mstack
            delta_Mstack = torch.clamp(delta_Mstack, min=0) #make sure it is positive
            delta_list=Mstack2list(delta_Mstack)
            delta_lists.append(delta_list)

        
            #multi-round iteration for theta
            S=50
            theta_list=theta_lists[t]
            s_list=[]
            #weights=torch.ones((self.J,self.J),dtype=torch.double)/self.J
            weights=self.weights
            #weights=torch.matrix_power(self.weights, 5)
            for s in range(S):

                if t==0 and s==0:
                    y_theta_Mstack=torch.tensordot(y_theta_f_parallel(mu_list,Sigma_list,beta_lists[1],delta_lists[1],theta_list), weights, dims=1)
                    y_hessian_theta_Mstack=torch.tensordot(y_hessian_theta_f_parallel(mu_list,Sigma_list,beta_lists[1],delta_lists[1],theta_list), weights, dims=1)
                elif t>=1 and s==0:
                    y_theta_Mstack=torch.tensordot(y_theta_Mstack+y_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list)-y_theta_f_parallel(mu_list_p,Sigma_list_p,beta_lists[t],delta_lists[t],theta_list_p), weights, dims=1)
                    y_hessian_theta_Mstack=torch.tensordot(y_hessian_theta_Mstack+y_hessian_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list)-y_hessian_theta_f_parallel(mu_list_p,Sigma_list_p,beta_lists[t],delta_lists[t],theta_list_p), weights, dims=1)
               
                else:
                    y_theta_Mstack=torch.tensordot(y_theta_Mstack+y_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list)-y_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list_p), weights, dims=1)
                    y_hessian_theta_Mstack=torch.tensordot(y_hessian_theta_Mstack+y_hessian_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list)-y_hessian_theta_f_parallel(mu_list,Sigma_list,beta_lists[t+1],delta_lists[t+1],theta_list_p), weights, dims=1)
                
                #there are some modifications
                #com_grad_theta_Mstack=com_grad_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                #com_hessian_theta_Mstack=com_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                com_grad_theta_Mstack=torch.tensordot(com_grad_theta_f_parallel(mu_list,Sigma_list,theta_list),weights, dims=1)
                com_hessian_theta_Mstack=torch.tensordot(com_hessian_theta_f_parallel(mu_list,Sigma_list,theta_list),weights, dims=1)
                grad_theta_Mstack=y_theta_Mstack*self.J+com_grad_theta_Mstack
                
                #print(torch.norm(torch.mean(grad_theta_Mstack,dim=1)))
                if s>=6 and torch.norm(torch.mean(grad_theta_Mstack,dim=1))<1e-4:
                    break
                hessian_theta_Mstack=y_hessian_theta_Mstack*self.J+com_hessian_theta_Mstack
               
                theta_Mstack=list2Mstack(theta_list)
                invh_m_grad_Mstack=torch.zeros_like(grad_theta_Mstack)
                
                noise=torch.randn(size=(2,1))*0.01
                for j in range(self.J):
                    hess=hessian_theta_Mstack[:,:,j]
                    grad=grad_theta_Mstack[:,j].unsqueeze(1)

                    eigenvalues, eigenvectors = torch.linalg.eigh(hess)
                    # Take the absolute value of each eigenvalue
                    abs_eigenvalues = eigenvalues.abs()

                    # Define a positive threshold
                    threshold = 0.01

                    # Replace elements smaller than the threshold with the threshold value
                    modified_eigenvalues = torch.where(abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)

                    # Construct the diagonal matrix of modified eigenvalues
                    modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)

                    modified_hess=eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
                    invh_m_grad_Mstack[:,j:(j+1)]=torch.linalg.inv(modified_hess)@grad
                    if torch.all(eigenvalues>0):
                        invh_m_grad_Mstack[:,j:(j+1)]=torch.linalg.inv(modified_hess)@grad
                    else:
                        invh_m_grad_Mstack[:,j:(j+1)]=0.1*torch.linalg.inv(modified_hess)@grad
                
                #set the step size via backtracking line search
                if s>=6 and torch.norm(torch.mean(invh_m_grad_Mstack,dim=1))<1e-5:
                    break
    
                step_size=1
                shrink_rate=0.5
                #m=0.1
                Continue=True
                # f_value=y_value_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list)*self.J+com_value_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                # theta_Mstack_new=torch.tensordot(theta_Mstack-step_size*invh_m_grad_Mstack, weights, dims=1) 
                # theta_list_new=Mstack2list(theta_Mstack_new)     
                # theta_list_p=theta_list.copy()
                # theta_list=theta_list_new
               
                while Continue:
                    theta_Mstack_new=torch.tensordot(theta_Mstack-step_size*invh_m_grad_Mstack, self.weights, dims=1)  
                    theta_list_new=Mstack2list(theta_Mstack_new) 
                    if torch.all(theta_Mstack_new>0.05):
                        
                        if s>=1 and step_size>0.01 and torch.norm(torch.mean(grad_theta_Mstack,dim=1))>=torch.norm(torch.mean(grad_theta_Mstack_p,dim=1)):
                            step_size*=shrink_rate
                        else:
                           theta_list_p=theta_list.copy()
                           theta_list=theta_list_new
                           Continue=False
                    else:
                        if step_size>0.01:
                            step_size*=shrink_rate
                        else:
                            theta_Mstack=theta_Mstack+noise
                            Continue=False 
                   
                        
                # while Continue:
                #     theta_Mstack_new=torch.tensordot(theta_Mstack-step_size*invh_m_grad_Mstack, self.weights, dims=1)  
                #     theta_list_new=Mstack2list(theta_Mstack_new) 
                #     if torch.all(theta_Mstack_new>0.005):
                #         f_value_new=y_value_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list_new)*self.J+com_value_f(mu_lists[t+1],Sigma_lists[t+1],theta_list_new)
                #         if f_value_new>f_value-m*step_size*torch.dot(torch.mean(grad_theta_Mstack, dim=1),torch.mean(invh_m_grad_Mstack, dim=1)):
                #         #if f_value_new>f_value+10:
                #             step_size*=shrink_rate
                #         else:
                #            theta_list_p=theta_list.copy()
                #            theta_list=theta_list_new
                #            Continue=False
                #     else:
                #         if step_size>1e-4:
                #             step_size*=shrink_rate
                #         else:
                #             theta_Mstack=theta_Mstack+noise
                #             Continue=False 
                #print(f"theta:{torch.mean(theta_Mstack,dim=1).numpy()},gradient:{torch.mean(grad_theta_Mstack,dim=1).numpy()},norm of grad:{torch.norm(torch.mean(grad_theta_Mstack,dim=1)).numpy()}")
                # print(torch.norm(torch.mean(grad_theta_Mstack,dim=1)))
                # if torch.norm(torch.mean(grad_theta_Mstack,dim=1))<1e-4:
                #     break
                if s>=6 and torch.norm(torch.mean(grad_theta_Mstack,dim=1)-torch.mean(grad_theta_Mstack_p,dim=1))<1e-4:
                    break
                grad_theta_Mstack_p=grad_theta_Mstack
            theta_lists.append(theta_list)
            s_list.append(s)


        return mu_list,Sigma_list,beta_lists,delta_lists,theta_lists,s_list


    def ce_optimize_stage2(self,mu:torch.Tensor,Sigma:torch.Tensor,beta:torch.Tensor,delta:torch.Tensor,theta:torch.Tensor,T:int,job_num,seed=2024,thread_num=None,backend='threading'):
        torch.manual_seed(seed)
        if thread_num!=None:
            torch.set_num_threads(thread_num)
        #define some functions
        def K_f(theta):
            return self.kernel(self.knots, self.knots, theta)
        
        def local_B_f(j,theta):
            data=self.dis_data[j]
            local_locs = data[:, :2]
            K = K_f(theta)
            invK=torch.linalg.inv(K)
            B = self.kernel(local_locs, self.knots, theta) @ invK
            return B
        
        def local_size_f(j):
            return self.dis_data[j].shape[0]
        
        def local_erorrV_f(j,beta):
            data=self.dis_data[j]
            local_z = data[:, 2].reshape(-1,1)      # Extract the third column as local z
            local_X = data[:, 3:]     # Extract columns from the fourth to the end as local X
            errorv=local_X@beta-local_z
            return errorv
        
        def local_X_f(j):
            data=self.dis_data[j]
            local_X = data[:, 3:]
            return local_X
        
        def local_z_f(j):
            data=self.dis_data[j]
            local_z = data[:, 2].reshape(-1,1)
            return local_z
        
        #parallel
        y_XTX=torch.zeros((self.p,self.p),dtype=torch.double)
        def compute_XTX_j(j):
            local_X = local_X_f(j)
            return local_X.T @ local_X
        results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_XTX_j)(j) for j in range(self.J))
        for j in range(self.J):
            y_XTX+=results[j]
        y_XTX=y_XTX/self.J
        
        
        # #non-parallel
        # y_XTX=torch.zeros((self.p,self.p),dtype=torch.double)
        # for j in range(self.J):
        #     local_X=local_X_f(j)
        #     y_XTX+=local_X.T@local_X
        # y_XTX=y_XTX/self.J
        
      
        def y_mu_f_parallel(beta, theta):
            y_mu_M = torch.zeros((self.m, 1), dtype=torch.double)
            
            def compute_j(j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                return -local_B.T @ local_errorV

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_j)(j, beta, theta) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_mu_M =y_mu_M+ results[j]
            y_mu_M=y_mu_M/self.J
            return y_mu_M
        
        
        def y_mu_f(beta,theta):
            y_mu_M=torch.zeros((self.m,1),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta)
                local_erorrV=local_erorrV_f(j,beta)
                y_mu_M+=-local_B.T@local_erorrV
            y_mu_M=y_mu_M/self.J
            return y_mu_M
        
        
       
        
        def y_Sigma_f_parallel(theta):
            y_Sigma_M = torch.zeros((self.m, self.m), dtype=torch.double)
            
            def compute_j(j, theta_j):
                local_B = local_B_f(j, theta_j)
                return local_B.T @ local_B

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_j)(j, theta) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_Sigma_M+= results[j]
            y_Sigma_M=y_Sigma_M/self.J
            return y_Sigma_M
        
        def y_Sigma_f(theta):
            y_Sigma_M=torch.zeros((self.m,self.m),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta)
                y_Sigma_M+=local_B.T@local_B
            y_Sigma_M=y_Sigma_M/self.J
            return y_Sigma_M
        
     
        
        def y_beta_f_parallel(mu, theta):
            y_beta_M = torch.zeros((self.p, 1), dtype=torch.double)
            
            def compute_j(j, mu_j, theta_j):
                local_X = local_X_f(j)
                local_B = local_B_f(j, theta_j)
                local_z = local_z_f(j)
                return local_X.T @ (local_z - local_B @ mu_j)

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_j)(j, mu, theta) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_beta_M+= results[j]
            y_beta_M=y_beta_M/self.J
            return y_beta_M
        
        def y_beta_f(mu,theta):
            y_beta_M=torch.zeros((self.p,1),dtype=torch.double)
            for j in range(self.J):
                local_X=local_X_f(j)
                local_B=local_B_f(j,theta)
                local_z=local_z_f(j)
                y_beta_M+=local_X.T@(local_z-local_B@mu)
            y_beta_M=y_beta_M/self.J
            return y_beta_M
      
        def y_delta_f_parallel(mu, Sigma, beta, theta):
            y_delta = torch.zeros((1,1), dtype=torch.double)
            
            def compute_j(j, mu_j, Sigma_j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                
                term1 = torch.trace(local_B.T @ local_B @ (Sigma_j + mu_j @ mu_j.T))
                term2 = 2 * local_errorV.T @ local_B @ mu_j
                term3 = local_errorV.T @ local_errorV
                
                return term1 + term2 + term3

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_j)(j, mu, Sigma, beta, theta) for j in range(self.J))
            
            # Stack the results
            for j in range(self.J):
                y_delta += results[j]
            y_delta=y_delta/self.J
            return y_delta
        
        def y_delta_f(mu,Sigma,beta,theta):
            y_delta=torch.zeros((1,1),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta)
                local_erorrV=local_erorrV_f(j,beta)
                y_delta+=torch.trace(local_B.T@local_B@(Sigma+mu@mu.T))+2*local_erorrV.T@local_B@mu+local_erorrV.T@local_erorrV
            y_delta=y_delta/self.J
            return y_delta
        
       
        def y_value_f_parallel(mu,Sigma,beta,delta,theta):
            y_value_M=torch.zeros((1,1),dtype=torch.double)
            def compute_j(j, mu_j, Sigma_j, beta_j,delta_j,theta_j):
                local_value=self.local_value(j,mu_j, Sigma_j, beta_j,delta_j, theta_j)
                return local_value
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_j)(j, mu,Sigma,beta,delta,theta) for j in range(self.J))
            for j in range(self.J):
                y_value_M+=results[j].reshape(-1,1)
            y_value_M=y_value_M/self.J
            return y_value_M
        
        def y_value_f(mu,Sigma,beta,delta,theta):
            y_value_M=torch.zeros((1,1),dtype=torch.double)
            for j in range(self.J):
                local_value=self.local_value(j,mu,Sigma,beta,delta,theta)
                y_value_M+=local_value.reshape(-1,1)
            y_value_M=y_value_M/self.J
            return y_value_M
    
        def y_theta_f_parallel(mu, Sigma, beta, delta, theta):
          
            y_theta_M = torch.zeros((2, 1), dtype=torch.double)
            
            
            knots=self.knots
            # Compute the kernel matrices
            K = self.kernel(knots, knots, theta)
            invK=torch.linalg.inv(K)
            
            def compute_local_grad(j):
                data=self.dis_data[j]
                local_locs = data[:, :2]  # Extract the first two columns as local locations
                local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
                local_X = data[:, 3:] 
                
                n = local_z.shape[0]
                
                theta=theta.clone()
                if theta.grad is not None:
                    theta.grad.data.zero_()
                theta.requires_grad_(True)
                B = self.kernel(local_locs, knots, theta) @ invK
                grad = theta.grad
                # Compute the value of the local objective function
                errorV= local_X @ beta-local_z
                f_value = -n * torch.log(delta) + delta * (
                    torch.trace(B.T @ B @ (Sigma + mu @ mu.T)) 
                    + 2 * errorV.T @ B @ mu 
                    + errorV.T @ errorV
                )        
                f_value.backward()
                return grad

            def compute_local_grad(j, mu_j, Sigma_j, beta_j, delta_j, theta_j):
                local_g_theta = self.local_grad_theta(j, mu_j, Sigma_j, beta_j, delta_j, theta_j)
                return j, local_g_theta.reshape(-1, 1)

            # Parallelize the computation using joblib
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_local_grad)(j, mu, Sigma, beta, delta, theta) for j in range(self.J))

            # Populate y_theta_Mstack with the results
            for j, local_g_theta in results:
                y_theta_M += local_g_theta
            y_theta_M=y_theta_M/self.J
            return y_theta_M  
        
        def y_theta_f(mu,Sigma,beta,delta,theta):
            y_theta_M=torch.zeros((2,1),dtype=torch.double)
            for j in range(self.J):
                local_g_theta=self.local_grad_theta(j,mu,Sigma,beta,delta,theta)
                y_theta_M+=local_g_theta.reshape(-1,1)
            y_theta_M=y_theta_M/self.J
            return y_theta_M
        
    
        
        def y_hessian_theta_f_parallel(mu, Sigma, beta, delta, theta):
            hessian_theta_M=torch.zeros((2,2),dtype=torch.double)
            def compute_local_hessian(j,mu_j, Sigma_j, beta_j, delta_j, theta_j):
                local_h_theta = self.local_hessian_theta(j, mu_j, Sigma_j, beta_j, delta_j, theta_j)
                return j, local_h_theta
            results = Parallel(n_jobs=job_num,backend=backend)(delayed(compute_local_hessian)(j, mu, Sigma, beta, delta, theta) for j in range(self.J))

            for j, local_h_theta in results:
                hessian_theta_M+=local_h_theta
            hessian_theta_M=hessian_theta_M/self.J
            return hessian_theta_M
        
        def y_hessian_theta_f(mu,Sigma,beta,delta,theta):
            hessian_theta_M=torch.zeros((2,2),dtype=torch.double)
            for j in range(self.J):
                local_h_theta=self.local_hessian_theta(j,mu,Sigma,beta,delta,theta)
                hessian_theta_M+=local_h_theta
            hessian_theta_M=hessian_theta_M/self.J
            return hessian_theta_M
        
        
        def com_value_f(mu,Sigma,theta):
            com_value_M=torch.zeros((1,1),dtype=torch.double)
            com_value=self.com_value(mu,Sigma,theta)            
            com_value_M=com_value.reshape(-1,1)
            return com_value_M
        
        def com_grad_theta_f(mu,Sigma,theta):
            com_grad_theta_M=torch.zeros((2,1),dtype=torch.double)
            com_g_theta=self.com_grad_theta(mu,Sigma,theta)            
            com_grad_theta_M=com_g_theta.reshape(-1,1)
            return com_grad_theta_M
        
      
        def com_hessian_theta_f(mu,Sigma,theta):
            com_hessian_theta_M=torch.zeros((2,2),dtype=torch.double)
            com_h_theta=self.com_hessian_theta(mu,Sigma,theta)            
            com_hessian_theta_M=com_h_theta
            return com_hessian_theta_M
        
        
        size=torch.zeros((1,1),dtype=torch.double) 
        for j in range(self.J):
            size+=local_size_f(j)
        size=size/self.J



      
        beta_list=[beta]
        delta_list=[delta]
        theta_list=[theta]
        s_list=[]
        for t in range(T):
            if t%10==0:
                print(f"iteration:{t}", end=', ')
            #print(f"iteration:{t}")
            #mu and Sigma
            y_mu=y_mu_f_parallel(beta_list[t],theta_list[t])
            y_Sigma=y_Sigma_f_parallel(theta_list[t])
            K=K_f(theta_list[t])
            invK=torch.linalg.inv(K)
            Sigma=torch.linalg.inv(delta_list[t]*self.J*y_Sigma+invK)
            mu=torch.linalg.inv(delta_list[t]*y_Sigma+invK/self.J)*delta_list[t]@y_mu

            #beta
            y_beta=y_beta_f_parallel(mu,theta_list[t])
            beta=torch.linalg.inv(y_XTX)@y_beta
            beta_list.append(beta)


            #delta
            y_delta=y_delta_f_parallel(mu,Sigma,beta_list[t+1],theta_list[t])
            delta=size/y_delta
            delta_list.append(delta)

            #multi-round iteration
            S=50
            theta=theta_list[t]
            for s in range(S):
                y_theta=y_theta_f_parallel(mu,Sigma,beta_list[t+1],delta_list[t+1],theta)
                y_hessian_theta=y_hessian_theta_f_parallel(mu,Sigma,beta_list[t+1],delta_list[t+1],theta)
                
                com_grad_theta=com_grad_theta_f(mu,Sigma,theta)
                com_hessian_theta=com_hessian_theta_f(mu,Sigma,theta)
                grad=y_theta*self.J+com_grad_theta

                if torch.norm(grad)<1e-4:
                    break

                hess=y_hessian_theta*self.J+com_hessian_theta
                eigenvalues, eigenvectors = torch.linalg.eigh(hess)

                # Take the absolute value of each eigenvalue
                abs_eigenvalues = eigenvalues.abs()

                # Define a positive threshold
                threshold = 0.01

                # Replace elements smaller than the threshold with the threshold value
                modified_eigenvalues = torch.where(abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)

                # Construct the diagonal matrix of modified eigenvalues
                modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)

                modified_hess=eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
                
                noise=torch.randn(size=(2,1))*0.01

                invh_m_grad=torch.linalg.inv(modified_hess)@grad
                if torch.norm(invh_m_grad)<1e-5:
                    break
                #set the step size via backtracking line search
                step_size=1
                shrink_rate=0.8
                m=0.1
                Continue=True
                f_value=y_value_f_parallel(mu,Sigma,beta_list[t+1],delta_list[t+1],theta)*self.J+com_value_f(mu,Sigma,theta)
               
                while Continue:
                    theta_new=theta-step_size*invh_m_grad
                    if torch.all(theta_new>0.005):
                        f_value_new=y_value_f_parallel(mu,Sigma,beta_list[t+1],delta_list[t+1],theta_new)*self.J+com_value_f(mu,Sigma,theta_new)
                        if f_value_new>f_value-m*step_size*torch.dot(grad.squeeze(),invh_m_grad.squeeze()):
                           step_size*=shrink_rate
                        else:
                           theta=theta_new
                           Continue=False
                    else:
                        if step_size>1e-4:
                            step_size*=shrink_rate
                        else:
                            theta=theta+noise
                            Continue=False

                
                #print(f"theta:{theta},gradient:{torch.norm(grad)}")
            theta_list.append(theta)
            s_list.append(s)

        return mu,Sigma,beta_list,delta_list,theta_list,s_list