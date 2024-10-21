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

class GPPPrediction:
    def __init__(self, location: torch.Tensor, kernel: callable, knots: torch.Tensor,X: torch.Tensor, mu: torch.Tensor,Sigma: torch.Tensor,beta: torch.Tensor,delta: torch.Tensor,theta: torch.Tensor) -> None:
        """
        Initialize the class with locations, a kernel function, knots, and parameters.
        """
        self.location = location  # locations
        self.knots = knots        # Knot points tensor
        self.m = len(self.knots)  # Number of knots
        self.kernel = kernel      # Kernel function
        self.X=X                  # covariates
        
        #parameters
        self.mu=mu   
        self.Sigma=Sigma
        self.beta=beta
        self.delta=delta
        self.theta=theta
        
        
    def predict(self):
        B=self.kernel(self.location,self.knots,self.theta)@torch.linalg.inv(self.kernel(self.knots,self.knots,theta=self.theta))
        mean_pred=torch.tensor(self.X)@self.beta+B@self.mu
        N_pre=self.location.shape[0]
        cov_pred=B@self.Sigma@B.T+1/(self.delta**2)*torch.eye(N_pre)
        
        return mean_pred,cov_pred