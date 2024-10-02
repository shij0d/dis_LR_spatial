from __future__ import annotations
import numpy as np
import random
from src.kernel import Kernel_with_Grad,Matern_with_Grad
from sklearn.gaussian_process.kernels import Matern

import numpy as np
import random
from typing import Callable, Any, List
from src.utils import softplus_torch,inv_softplus_torch,replace_negative_eigenvalues_with_zero,softplus_d
from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian,jacobian

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
    
    def com_grad_theta(self,mu:torch.Tensor,Sigma:torch.Tensor,theta:torch.Tensor):
        theta=theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))

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

    def neg_log_lik(self, local_locs:torch.Tensor, local_z:torch.Tensor, local_X:torch.Tensor, params:torch.Tensor, requires_grad=True,requires_hess=False):
      
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
            grad = torch.autograd.grad(f_value, params, create_graph=True)[0]
            if requires_hess:
                # Compute the Hessian
                hessian = torch.zeros(params.size(0), params.size(0))
                for i in range(params.size(0)):
                    grad_i = grad[i]
                    hessian[i] = torch.autograd.grad(grad_i, params, retain_graph=True)[0]
                return f_value,grad,hessian
            else:
                return f_value,grad
        else:
            return f_value
        # if requires_grad:
        #     # Compute the gradients
        #     f_value.backward()
        #     grad = params.grad
        #     return f_value, grad
        # else:
        #     return f_value
 
    def local_neg_log_lik_wrapper(self,j,requires_grad=True,requires_hess=False):
        """
        the negative local log likelihood function for the low rank model 
        """

        data=self.dis_data[j]
        local_locs = data[:, :2]  # Extract the first two columns as local locations
        local_z = data[:, 2].unsqueeze(1)      # Extract the third column as local z
        local_X = data[:, 3:]
        if requires_grad:
            if requires_hess:
                def fun(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    value=self.neg_log_lik(local_locs,local_z,local_X,params,False)
                    return value.numpy().flatten()
                def gradf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,grad=self.neg_log_lik(local_locs,local_z,local_X,params,True)
                    return grad.detach().numpy()
                def hessf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,_,hess=self.neg_log_lik(local_locs,local_z,local_X,params,True,True)
                    return hess.numpy()
                
                return fun,gradf,hessf
            else:
                def fun(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    value=self.neg_log_lik(local_locs,local_z,local_X,params,False)
                    return value.numpy().flatten()
                def gradf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,grad=self.neg_log_lik(local_locs,local_z,local_X,params,True)
                    return grad.detach().numpy()
                
                return fun,gradf
        else:
            
            def fun(params:np.ndarray):
                params=torch.tensor(params,dtype=torch.float64)
                value=self.neg_log_lik(local_locs,local_z,local_X,params,False)
                return value.numpy().flatten()
            return fun
    def neg_log_lik_wrapper(self,requires_grad=True,requires_hess=False):
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
            if requires_hess:
                def fun(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    value=self.neg_log_lik(locs,z,X,params,False)
                    return value.numpy().flatten()
                def gradf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,grad=self.neg_log_lik(locs,z,X,params,True)
                    return grad.detach().numpy()
                def hessf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,_,hess=self.neg_log_lik(locs,z,X,params,True,True)
                    return hess.numpy()
                
                return fun,gradf,hessf
            else:
                def fun(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    value=self.neg_log_lik(locs,z,X,params,False)
                    return value.numpy().flatten()
                def gradf(params:np.ndarray):
                    params=torch.tensor(params,dtype=torch.float64)
                    _,grad=self.neg_log_lik(locs,z,X,params,True)
                    return grad.detach().numpy()
                
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
    
    def get_local_minimizer(self,j:int,x0:torch.Tensor):
        """
        Optimize the local likelihood functions in each machine to obtain the initial points
        """
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
    def get_local_minimizers(self,x0:torch.Tensor):
        mu_list=[]
        Sigma_list=[]
        beta_list=[]
        delta_list=[]
        theta_list=[]
        result_list=[]
        for j in range(self.J):
            mu,Sigma,beta, delta, theta,result=self.get_local_minimizer(j,x0)
            mu_list.append(mu)
            Sigma_list.append(Sigma)
            beta_list.append(beta)
            theta_list.append(theta)
            delta_list.append(delta)
            result_list.append(result)
        return mu_list,Sigma_list,beta_list,delta_list,theta_list,result_list
    def get_minimier(self,x0:torch.Tensor):
        x0=x0.numpy()
        nllikf,nllikgf,nllikhf=self.neg_log_lik_wrapper(requires_grad=True,requires_hess=True)  
        result=minimize(fun=nllikf,
                     x0=x0,
                     method="BFGS",
                     jac=nllikgf,tol=1e-8)
        minimizer_lik=torch.tensor(result.x,dtype=torch.float64)
        mu,Sigma=self.get_pos(minimizer_lik)
        beta, delta, theta=self.vector2arguments_lik(minimizer_lik)
        hess=nllikhf(minimizer_lik)
        return (mu,Sigma,beta, delta, theta,result,hess)
    

    def dis_opimize_stage1(self,initial_params:torch.Tensor,T:int):
        step_size=0.1
        epsilon=0.01
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
            
            
            step_size_g=torch.ones(length)*2
            step_size_g[-2]=1000
            step_size_g[-1]=20
            step_size_qn=1
            step_size_bb=0.01
            noise=torch.randn(size=(length,1))*epsilon/length
            
                #y_local_grad=weights_kron_f(length)@(y_local_grad+local_grads(params_Mstack_list[t])-local_grads(params_Mstack_list[t-1]))
            if t==0:
                grad=(sum_grad(params)+com_grad(params))/10**4
                grad.squeeze_()
                grad_noise=grad#+noise
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
                print(f"An unexpected error occurred: {e}")
                exception_occurred_bb = True
                
            grad_norm_g=torch.norm(grad_g)
            grad_norm_bb=torch.norm(grad_bb)
            
            if exception_occurred_bb==True or grad_norm_g<grad_norm_bb:
                params_new=params_g
                grad_new=grad_g
                print("gradient descent method")
            else:
                params_new=params_bb
                grad_new=grad_bb
                print("BB")
            # params_new=params_bb
            # grad_new=grad_bb
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
            print("gamma_t0:",torch.dot(y_t, y_t) / torch.dot(y_t, s_t))
            h_t0=1/gamma_t0

            temp1=torch.dot(y_t, s_t)
            temp2=torch.dot(s_t, s_t)/h_t0

            if temp1<0.25* temp2:
                zeta_t=0.75* temp2/(temp2-temp1)
            else:
                zeta_t=1
            zeta_t=1    
            print("zeta_t:",zeta_t)
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
            print(f'Iteration {t}, theta = {theta}, grad_norm = {grad_norm}')

            # if grad_new_norm<grad_norm*0.5:
            #     step_size=step_size*1.5
            # elif grad_new_norm>grad_norm:
            #     step_size=step_size*0.5
            # print(step_size)
            params=params_new
            grad=grad_new
            grad_noise=grad#+noise
            #params=params-torch.mul(grad_noise,step_size)
            params_list.append(params)
            # if t%10==0:
            #     print("theta,",theta.numpy())
            #     print(grad[5156:5158,0].numpy())
            
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
    
    def de_optimize_stage2(self,mu_list:list[torch.Tensor],Sigma_list:list[torch.Tensor],beta_list:list[torch.Tensor],delta_list:list[torch.Tensor],theta_list:list[torch.Tensor],T:int):
       
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
        
        y_XTX_Mstack=torch.zeros((self.J*self.p,self.p),dtype=torch.double)
        for j in range(self.J):
            local_X=local_X_f(j)
            y_XTX_Mstack[(self.p*j):(self.p*(j+1)),:]=local_X.T@local_X
        
        def y_mu_f(beta_list,theta_list):
            y_mu_Mstack=torch.zeros((self.J*self.m,1),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                local_erorrV=local_erorrV_f(j,beta_list[j])
                y_mu_Mstack[(self.m*j):(self.m*(j+1)),:]=-local_B.T@local_erorrV
            return y_mu_Mstack
        
        def y_Sigma_f(theta_list):
            y_Sigma_Mstack=torch.zeros((self.J*self.m,self.m),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                y_Sigma_Mstack[(self.m*j):(self.m*(j+1)),:]=local_B.T@local_B
            return y_Sigma_Mstack
        
        def y_gbeta_f(mu_list,beta_list,theta_list):
            y_beta_Mstack=torch.zeros((self.J*self.p,1),dtype=torch.double)
            for j in range(self.J):
                local_X=local_X_f(j)
                local_B=local_B_f(j,theta_list[j])
                local_erorrV=local_erorrV_f(j,beta_list[j])
                y_beta_Mstack[(self.p*j):(self.p*(j+1)),:]=local_X.T@(local_erorrV+local_B@mu_list[j])
            return y_beta_Mstack
        
        def y_beta_f(mu_list,theta_list):
            y_beta_Mstack=torch.zeros((self.J*self.p,1),dtype=torch.double)
            for j in range(self.J):
                local_X=local_X_f(j)
                local_B=local_B_f(j,theta_list[j])
                local_z=local_z_f(j)
                y_beta_Mstack[(self.p*j):(self.p*(j+1)),:]=local_X.T@(local_z-local_B@mu_list[j])
            return y_beta_Mstack
        
        def y_delta_f(mu_list,Sigma_list,beta_list,theta_list):
            y_delta_Mstack=torch.zeros((self.J,1),dtype=torch.double)
            for j in range(self.J):
                local_B=local_B_f(j,theta_list[j])
                local_erorrV=local_erorrV_f(j,beta_list[j])
                y_delta_Mstack[j,:]=torch.trace(local_B.T@local_B@(Sigma_list[j]+mu_list[j]@mu_list[j].T))+2*local_erorrV.T@local_B@mu_list[j]+local_erorrV.T@local_erorrV
            return y_delta_Mstack
        
        def y_theta_f(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            y_theta_Mstack=torch.zeros((self.J*2,1),dtype=torch.double)
            for j in range(self.J):
                local_g_theta=self.local_grad_theta(j,mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
                y_theta_Mstack[(2*j):(2*(j+1)),:]=local_g_theta.reshape(-1,1)
            return y_theta_Mstack
        
        def y_hessian_theta_f(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            hessian_theta_Mstack=torch.zeros((self.J*2,2),dtype=torch.double)
            for j in range(self.J):
                local_h_theta=self.local_hessian_theta(j,mu_list[j],Sigma_list[j],beta_list[j],delta_list[j],theta_list[j])
                hessian_theta_Mstack[(2*j):(2*(j+1)),:]=local_h_theta
            return hessian_theta_Mstack

        def com_grad_theta_f(mu_list,Sigma_list,theta_list):
            com_grad_theta_Mstack=torch.zeros((self.J*2,1),dtype=torch.double)
            for j in range(self.J):
                com_g_theta=self.com_grad_theta(mu_list[j],Sigma_list[j],theta_list[j])            
                com_grad_theta_Mstack[(2*j):(2*(j+1)),:]=com_g_theta.reshape(-1,1)
            return com_grad_theta_Mstack
        
        def com_hessian_theta_f(mu_list,Sigma_list,theta_list):
            com_hessian_theta_Mstack=torch.zeros((self.J*2,2),dtype=torch.double)
            for j in range(self.J):
                com_h_theta=self.com_hessian_theta(mu_list[j],Sigma_list[j],theta_list[j])            
                com_hessian_theta_Mstack[(2*j):(2*(j+1)),:]=com_h_theta
            return com_hessian_theta_Mstack
        
        def weights_kron_f(num,iter=1):
            weight=self.weights
            for i in range(iter):
                weight=weight@weight
            weights_kron=torch.kron(weight,torch.eye(num,dtype=torch.double))
            return weights_kron
        
        def list2Mstack(lt:list[torch.Tensor]):
            J=len(lt)
            
            if lt[0].dim()==0:
                r=1
                c=1
            elif lt[0].dim()==1:
                r=lt[0].shape[0]
                c=1
            else:
                r=lt[0].shape[0]
                c=lt[0].shape[1]
            Mstack=torch.zeros((J*r,c),dtype=torch.double)
            for j in range(J):
                Mstack[(r*j):(r*(j+1)),:]=lt[j]
            return Mstack
        
        def Mstack2list(Mstack:torch.Tensor,r:int):
            lt=[]
            N=Mstack.shape[0]
            for j in range(N//r):
                lt.append(Mstack[(r*j):(r*(j+1)),:])
            return lt
        
        def avg_local_minimizer(mu_list,Sigma_list,beta_list,delta_list,theta_list):
            mu_Mstack=list2Mstack(mu_list)
            Sigma_Mstack=list2Mstack(Sigma_list)
            beta_Mstack=list2Mstack(beta_list)
            delta_Mstack=list2Mstack(delta_list)
            theta_Mstack=list2Mstack(theta_list)
            mu_Mstack=weights_kron_f(self.m)@mu_Mstack
            Sigma_Mstack=weights_kron_f(self.m)@Sigma_Mstack
            beta_Mstack=weights_kron_f(self.p)@beta_Mstack
            delta_Mstack=self.weights@delta_Mstack
            theta_Mstack=weights_kron_f(2)@theta_Mstack

            mu_list=Mstack2list(mu_Mstack,self.m)
            Sigma_list=Mstack2list(Sigma_Mstack,self.m)
            beta_list=Mstack2list(beta_Mstack,self.p)
            delta_list=Mstack2list(delta_Mstack,1)
            theta_list=Mstack2list(theta_Mstack,2)
            return mu_list,Sigma_list,beta_list,delta_list,theta_list
            

        size_stack=torch.zeros((self.J,1),dtype=torch.double) 
        for j in range(self.J):
            size_stack[j,:]=local_size_f(j)

        mu_list,Sigma_list,beta_list,delta_list,theta_list=avg_local_minimizer(mu_list,Sigma_list,beta_list,delta_list,theta_list)


        mu_lists=[mu_list]
        Sigma_lists=[Sigma_list]
        beta_lists=[beta_list]
        delta_lists=[delta_list]
        theta_lists=[theta_list]
        
        for t in range(T):
            print(f"iteration:{t}")
            #mu and Sigma
            if t==0:
                y_mu_Mstack=weights_kron_f(self.m,1)@y_mu_f(beta_lists[0],theta_lists[0])
                y_Sigma_Mstack=weights_kron_f(self.m,1)@y_Sigma_f(theta_lists[0])
            else: 
                y_mu_Mstack=weights_kron_f(self.m,1)@(y_mu_Mstack+y_mu_f(beta_lists[t],theta_lists[t])-y_mu_f(beta_lists[t-1],theta_lists[t-1]))
                y_Sigma_Mstack=weights_kron_f(self.m,1)@(y_Sigma_Mstack+y_Sigma_f(theta_lists[t])-y_Sigma_f(theta_lists[t-1]))
            
            mu_list=[]
            Sigma_list=[]
            for j in range(self.J):
                y_Sigma=y_Sigma_Mstack[(self.m*j):(self.m*(j+1)),:]
                y_Sigma=replace_negative_eigenvalues_with_zero(y_Sigma) #make sure it is positive definite
                y_mu=y_mu_Mstack[(self.m*j):(self.m*(j+1)),:]
                K=K_f(theta_lists[t][j])
                invK=torch.linalg.inv(K)
                Sigma=torch.linalg.inv(delta_lists[t][j]*self.J*y_Sigma+invK)
                mu=torch.linalg.inv(delta_lists[t][j]*y_Sigma+invK/self.J)*delta_lists[t][j]@y_mu
                mu_list.append(mu)
                Sigma_list.append(Sigma)
            mu_lists.append(mu_list)
            Sigma_lists.append(Sigma_list)


            #beta

            ##gradient desent
            # if t==0:
            #     y_gbeta_Mstack=weights_kron_f(self.p)@y_gbeta_f(mu_lists[1],beta_lists[0],theta_lists[0])
            # else:
            #     y_gbeta_Mstack=weights_kron_f(self.p)@(y_gbeta_Mstack+y_gbeta_f(mu_lists[t+1],beta_lists[t],theta_lists[t])-y_gbeta_f(mu_lists[t],beta_lists[t-1],theta_lists[t-1]))
            # beta_Mstack=list2Mstack(beta_lists[t])
            # beta_Mstack=weights_kron_f(self.p)@(beta_Mstack-step_size_beta*y_gbeta_Mstack) # how to set the step size
            # beta_list=Mstack2list(beta_Mstack,self.p)
            # beta_lists.append(beta_list)

            #exact solution
            y_XTX_Mstack=weights_kron_f(self.p)@y_XTX_Mstack
            if t==0:
                y_beta_Mstack=weights_kron_f(self.p)@y_beta_f(mu_lists[1],theta_lists[0])
            else:
                y_beta_Mstack=weights_kron_f(self.p)@(y_beta_Mstack+y_beta_f(mu_lists[t+1],theta_lists[t])-y_beta_f(mu_lists[t],theta_lists[t-1]))
            beta_list=[]
            for j in range(self.J):
                y_XTX=y_XTX_Mstack[(self.p*j):(self.p*(j+1)),:]
                y_beta=y_beta_Mstack[(self.p*j):(self.p*(j+1)),:]
                beta=torch.linalg.inv(y_XTX)@y_beta
                beta_list.append(beta)
            beta_lists.append(beta_list)


            #delta
            if t==0:
                y_delta_Mstack=weights_kron_f(1)@y_delta_f(mu_lists[1],Sigma_lists[1],beta_lists[1],theta_lists[0])
            else:
                y_delta_Mstack=weights_kron_f(1)@(y_delta_Mstack+y_delta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],theta_lists[t])-y_delta_f(mu_lists[t],Sigma_lists[t],beta_lists[t],theta_lists[t-1]))
            size_stack=weights_kron_f(1)@size_stack

            delta_Mstack=size_stack/y_delta_Mstack
            delta_Mstack = torch.clamp(delta_Mstack, min=0) #make sure it is positive
            delta_list=Mstack2list(delta_Mstack,1)
            delta_lists.append(delta_list)

            #theta
            
            #one round iteration
            # if t==0:
            #     y_theta_Mstack=weights_kron_f(2)@y_theta_f(mu_lists[1],Sigma_lists[1],beta_lists[1],delta_lists[1],theta_lists[0])
            # else:
            #     y_theta_Mstack=weights_kron_f(2)@(y_theta_Mstack+y_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_lists[t])-y_theta_f(mu_lists[t],Sigma_lists[t],beta_lists[t],delta_lists[t],theta_lists[t-1]))
            # com_grad_theta_Mstack=com_grad_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_lists[t])
            # grad_theta_Mstack=y_theta_Mstack*self.J+com_grad_theta_Mstack

            # theta_Mstack=list2Mstack(theta_lists[t])
            # theta_Mstack=weights_kron_f(2)@(theta_Mstack-step_size_theta*grad_theta_Mstack) #how to set this value
            # theta_Mstack = torch.clamp(theta_Mstack, min=0) #make sure it is positive
            # theta_list=Mstack2list(theta_Mstack,2)
            # theta_lists.append(theta_list)

            #multi-round iteration
            S=50
            theta_list=theta_lists[t]
            for s in range(S):
                if t==0 and s==0:
                    y_theta_Mstack=weights_kron_f(2,9)@y_theta_f(mu_lists[1],Sigma_lists[1],beta_lists[1],delta_lists[1],theta_list)
                    y_hessian_theta_Mstack=weights_kron_f(2,9)@y_hessian_theta_f(mu_lists[1],Sigma_lists[1],beta_lists[1],delta_lists[1],theta_list)
                elif t>=1 and s==0:
                    y_theta_Mstack=weights_kron_f(2,9)@(y_theta_Mstack+y_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list)-y_theta_f(mu_lists[t],Sigma_lists[t],beta_lists[t],delta_lists[t],theta_list_p))
                    y_hessian_theta_Mstack=weights_kron_f(2,9)@(y_hessian_theta_Mstack+y_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list)-y_hessian_theta_f(mu_lists[t],Sigma_lists[t],beta_lists[t],delta_lists[t],theta_list_p))
                else:
                    y_theta_Mstack=weights_kron_f(2,9)@(y_theta_Mstack+y_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list)-y_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list_p))
                    y_hessian_theta_Mstack=weights_kron_f(2,9)@(y_hessian_theta_Mstack+y_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list)-y_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],beta_lists[t+1],delta_lists[t+1],theta_list_p))
                com_grad_theta_Mstack=com_grad_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                com_hessian_theta_Mstack=com_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                grad_theta_Mstack=y_theta_Mstack*self.J+com_grad_theta_Mstack
                hessian_theta_Mstack=y_hessian_theta_Mstack*self.J+com_hessian_theta_Mstack
                
                theta_Mstack=list2Mstack(theta_list)
                invh_m_grad_Mstack=torch.zeros_like(grad_theta_Mstack)
                for j in range(self.J):
                    hess=hessian_theta_Mstack[(2*j):(2*(j+1)),:]
                    grad=grad_theta_Mstack[(2*j):(2*(j+1)),:]

                    eigenvalues, eigenvectors = torch.linalg.eigh(hess)
                    # Take the absolute value of each eigenvalue
                    abs_eigenvalues = eigenvalues.abs()

                    # Define a positive threshold
                    threshold = 1

                    # Replace elements smaller than the threshold with the threshold value
                    modified_eigenvalues = torch.where(abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)

                    # Construct the diagonal matrix of modified eigenvalues
                    modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)

                    modified_hess=eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
                    if torch.all(eigenvalues>0):
                        invh_m_grad_Mstack[(2*j):(2*(j+1)),:]=0.1*torch.linalg.inv(modified_hess)@grad
                    else:
                        invh_m_grad_Mstack[(2*j):(2*(j+1)),:]=0.1*torch.linalg.inv(modified_hess)@grad
                        # step_size=0.01
                       
                        # invh_m_grad_Mstack[(2*j):(2*(j+1)),:]=step_size*grad/modified_eigenvalues.reshape(-1,1)
                    
                theta_Mstack=weights_kron_f(2,9)@(theta_Mstack-invh_m_grad_Mstack)
                ## gradient 
                # else: 
                #     step_size=0.01
                #     theta_Mstack=weights_kron_f(2)@(theta_Mstack-step_size*grad_theta_Mstack)
                theta_list_p=theta_list.copy()
                theta_list=Mstack2list(theta_Mstack,2)
                print("grad_theta_Mstack[0:2]:",grad_theta_Mstack[0:2])
                if torch.norm(grad_theta_Mstack[0:2])<1e-4:
                    break
                if s>=1:
                    if torch.norm(grad_theta_Mstack[0:2]-grad_theta_Mstack_p[0:2])<1e-2:
                        break
                grad_theta_Mstack_p=grad_theta_Mstack
            print("theta_list[0]:",theta_list[0])
            theta_lists.append(theta_list)


        return mu_lists,Sigma_lists,beta_lists,delta_lists,theta_lists
