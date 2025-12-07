from __future__ import annotations
import numpy as np
import math
import numpy as np
from typing import Callable, Any, List
from src.utils import softplus_torch, inv_softplus_torch, replace_negative_eigenvalues_with_zero, softplus_d
from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian
from torch.autograd import grad
from joblib import Parallel, delayed

torch.autograd.set_detect_anomaly(True)


class GPPEstimation:
    def __init__(self, dis_data: list[torch.Tensor | np.ndarray], kernel: callable, knots: torch.Tensor | np.ndarray, weights: torch.Tensor | np.ndarray) -> None:
        """
        Initialize the class with distributed data, a kernel function, knots, and weights.

        Parameters:
        dis_data (list[torch.tensor]): List of tensors, each representing local data.
        kernel (callable): Kernel function to be used.
        knots (torch.tensor): Tensor representing knot points.
        weights (torch.tensor): Tensor representing weights.
        """
        for i in range(len(dis_data)):
            if not isinstance(dis_data[i], torch.Tensor):
                dis_data[i] = torch.tensor(dis_data[i], dtype=torch.float64)

        if not isinstance(knots, torch.Tensor):
            knots = torch.tensor(knots, dtype=torch.float64)

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float64)

        self.dis_data = dis_data  # List of distributed data tensors
        self.J = len(dis_data)    # Number of distributed data points
        self.knots = knots        # Knot points tensor
        self.m = len(self.knots)  # Number of knots
        self.p = dis_data[0].shape[1] - 3  # Dimensionality of the data minus 3
        self.weights = weights    # Weights tensor
        self.kernel = kernel      # Kernel function

    def vector2arguments(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        L_ol[indices[0], indices[1]] = params[start:(
            start + self.m * (self.m + 1) // 2)]
        start += self.m * (self.m + 1) // 2

        # Setting the diagonal elements to be the softplus of their original values
        L = L_ol.clone()
        diag_indices = torch.arange(self.m)
        L[diag_indices, diag_indices] = softplus_torch(
            L[diag_indices, diag_indices])
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

    def argument2vector(self, mu: torch.Tensor, Sigma: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        params = torch.empty(self.m+self.m*(self.m+1)//2 +
                             self.p+1+theta.shape[0], dtype=torch.float64)
        start = 0
        params[:self.m] = mu.squeeze()
        start += self.m
        L = torch.linalg.cholesky(Sigma)
        L_ol = L.clone()
        diag_indices = torch.arange(self.m)
        L_ol[diag_indices, diag_indices] = inv_softplus_torch(
            L_ol[diag_indices, diag_indices])
        tri_indices = torch.tril_indices(
            row=L_ol.shape[0], col=L_ol.shape[1], offset=0)
        params[start:(start + self.m * (self.m + 1) // 2)
               ] = L_ol[tri_indices[0], tri_indices[1]]
        start += self.m * (self.m + 1) // 2

        params[start:(start+self.p)] = beta.squeeze()
        start += self.p

        params[start] = inv_softplus_torch(delta)
        start += 1

        params[start:] = inv_softplus_torch(theta).squeeze()

        return params

    def vector2arguments_lik(self, params: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        start = 0
        if self.p > 0:
            # Extracting beta from params
            beta = params[start:start + self.p].unsqueeze(1)
            start += self.p
        else:
            beta = None

        # Extracting and transforming delta from params
        delta_ol = params[start]
        delta = softplus_torch(delta_ol)
        start += 1

        # Extracting and transforming theta values from params
        theta_ol = params[start:]
        theta = softplus_torch(theta_ol).unsqueeze(1)

        return (beta, delta, theta)

    def argument2vector_lik(self, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:

        params = torch.empty(self.p+1+theta.shape[0], dtype=torch.float64)

        start = 0
        if self.p > 0:
            params[start:(start+self.p)] = beta.squeeze()
            start += self.p

        params[start] = inv_softplus_torch(delta)
        start += 1

        params[start:] = inv_softplus_torch(theta).squeeze()
        return params

    def local_fun(self, local_locs: torch.Tensor, local_z: torch.Tensor, local_X: torch.Tensor, params: torch.Tensor | np.array, requires_grad=True):
        """
        Compute the local objective function value and optionally its gradient.

        local_locs: 2D array
        local_z: 1D array
        local_X: 2D array
        params: 1D array
        requires_grad: Boolean to indicate if gradient computation is needed
        """
        # Convert input data to torch tensors

        knots = self.knots
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
        errorV = local_X @ beta-local_z
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

    def local_fun_wrapper(self, j: int, requires_grad=True):
        """
        Local objectives in each machine.

        Returns:
            List[Callable]: A list of local objective functions.
        """
        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].unsqueeze(1)
        # Extract columns from the fourth to the end as local X
        local_X = data[:, 3:]
        if requires_grad:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.local_fun(
                    local_locs, local_z, local_X, params, False)
                return value.numpy().flatten()

            def gradf(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                _, grad = self.local_fun(
                    local_locs, local_z, local_X, params, True)
                return grad.numpy()
            return fun, gradf
        else:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.local_fun(
                    local_locs, local_z, local_X, params, False)
                return value.numpy().flatten()
            return fun

    def local_value(self, j, mu: torch.Tensor, Sigma: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor):
        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].unsqueeze(1)
        if self.p > 0:
            local_X = data[:, 3:]
        knots = self.knots
        n = local_z.shape[0]

        theta = theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)

        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        B = self.kernel(local_locs, knots, theta) @ torch.linalg.inv(K)

        # Compute the value of the local objective function
        if self.p > 0:
            errorV = local_X @ beta-local_z
        else:
            errorV = -local_z
        f_value = -n * torch.log(delta) + delta * (
            torch.trace(B.T @ B @ (Sigma + mu @ mu.T))
            + 2 * errorV.T @ B @ mu
            + errorV.T @ errorV
        )
        return f_value

    def local_grad_theta(self, j, mu: torch.Tensor, Sigma: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor):
        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].unsqueeze(1)
        if self.p > 0:
            local_X = data[:, 3:]
        knots = self.knots
        n = local_z.shape[0]

        theta = theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)

        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        B = self.kernel(local_locs, knots, theta) @ torch.linalg.inv(K)

        # Compute the value of the local objective function
        if self.p > 0:
            errorV = local_X @ beta-local_z
        else:
            errorV = -local_z
        f_value = -n * torch.log(delta) + delta * (
            torch.trace(B.T @ B @ (Sigma + mu @ mu.T))
            + 2 * errorV.T @ B @ mu
            + errorV.T @ errorV
        )
        f_value.backward()
        grad = theta.grad
        return grad

    def local_hessian_theta(self, j, mu: torch.Tensor, Sigma: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor):

        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].unsqueeze(1)
        if self.p > 0:
            local_X = data[:, 3:]
        knots = self.knots
        n = local_z.shape[0]
        if self.p > 0:
            errorV = local_X @ beta-local_z
        else:
            errorV = -local_z

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
            f_value = f_value.squeeze()
            return f_value

        Hessian = hessian(local_f, theta.squeeze())

        return Hessian

    def com_fun(self, params: torch.Tensor, requires_grad=True):
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
        f_value = mu.T @ invK @ mu + \
            torch.trace(invK @ Sigma) - torch.logdet(Sigma)+torch.logdet(K)

        if requires_grad:
            # Compute the gradients
            f_value.backward()
            grad = params.grad
            return f_value, grad
        else:
            return f_value

    def com_fun_wrapper(self, requires_grad=True) -> List[Callable]:
        """
        Wrapper function to generate callable functions for com_fun.

        Args:
            requires_grad (bool): Whether to include gradient functions. Default is True.

        Returns:
            List[Callable]: A list of callable functions.
        """
        if requires_grad:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.com_fun(params, False)
                return value.numpy().flatten()

            def gradf(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                _, grad = self.com_fun(params, True)
                return grad.numpy()
            return fun, gradf
        else:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.com_fun(params, False)
                return value.numpy().flatten()
            return fun

    def com_value(self, mu: torch.Tensor, Sigma: torch.Tensor, theta: torch.Tensor):
        theta = theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        f_value = mu.T @ invK @ mu + \
            torch.trace(invK @ Sigma) - torch.logdet(Sigma)+torch.logdet(K)

        return f_value

    def com_grad_theta(self, mu: torch.Tensor, Sigma: torch.Tensor, theta: torch.Tensor):
        theta = theta.clone()
        if theta.grad is not None:
            theta.grad.data.zero_()
        theta.requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.inverse(K)
        # f_value = mu.T @ invK @ mu + torch.trace(invK @ Sigma) - torch.log(torch.det(Sigma) / torch.det(K))
        f_value = mu.T @ invK @ mu + \
            torch.trace(invK @ Sigma) + torch.logdet(K)
        f_value.backward()
        grad = theta.grad
        return grad

    def com_hessian_theta(self, mu: torch.Tensor, Sigma: torch.Tensor, theta: torch.Tensor):

        def com_f(theta_v: torch.Tensor):
            K = self.kernel(self.knots, self.knots, theta_v)
            invK = torch.inverse(K)
            f_value = mu.T @ invK @ mu + \
                torch.trace(invK @ Sigma)+torch.logdet(K)
            return f_value
        Hessian = hessian(com_f, theta.squeeze())
        return Hessian

    def neg_log_lik(self, local_locs: torch.Tensor, local_z: torch.Tensor, local_X: torch.Tensor | None, params: torch.Tensor, requires_grad=True):

        knots = self.knots
        n = local_z.shape[0]

        # Convert params to torch tensor with gradient tracking
        params.requires_grad_(requires_grad)
        if params.grad is not None:
            params.grad.data.zero_()

        beta, delta, theta = self.vector2arguments_lik(params)

        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        invK = torch.linalg.inv(K)
        B = self.kernel(local_locs, knots, theta) @ invK

        tempM = invK+delta*B.T@B
        if local_X != None or beta != None:
            errorv = local_X@beta-local_z
        else:
            errorv = -local_z

        f_value = torch.logdet(tempM)+torch.logdet(K)-n*torch.log(delta)+delta*(
            errorv.T)@errorv-delta**2*(errorv.T@B@torch.linalg.inv(tempM)@B.T@errorv)
        f_value = f_value/n

        if requires_grad:
            # Compute the gradients
            f_value.backward()
            grad = params.grad
            return f_value, grad
        else:
            return f_value

    def local_neg_log_lik_wrapper(self, j, requires_grad=True):
        """
        the negative local log likelihood function for the low rank model
        """

        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].unsqueeze(1)
        if self.p > 0:
            local_X = data[:, 3:]
        else:
            local_X = None
        if requires_grad:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.neg_log_lik(
                    local_locs, local_z, local_X, params, False)
                return value.numpy().flatten()

            def gradf(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                _, grad = self.neg_log_lik(
                    local_locs, local_z, local_X, params, True)
                return grad.numpy()

            return fun, gradf
        else:

            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.neg_log_lik(
                    local_locs, local_z, local_X, params, False)
                return value.numpy().flatten()
            return fun

    def neg_log_lik_wrapper(self, requires_grad=True):
        """
        the negative local log likelihood function for the low rank model
        """
        local_locs_list = []
        local_z_list = []
        local_X_list = []
        N = 0
        for data in self.dis_data:
            # Extract the first two columns as local locations
            local_locs = data[:, :2]
            # Extract the third column as local z
            local_z = data[:, 2].unsqueeze(1)
            local_locs_list.append(local_locs)
            local_z_list.append(local_z)

            # X
            if self.p > 0:
                # Extract columns from the fourth to the end as local X
                local_X = data[:, 3:]
                local_X_list.append(local_X)
            n = len(local_z)

            N += n
        locs = torch.cat(local_locs_list, dim=0)
        z = torch.cat(local_z_list, dim=0)
        if self.p > 0:
            X = torch.cat(local_X_list, dim=0)
        else:
            X = None

        if requires_grad:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.neg_log_lik(locs, z, X, params, False)
                return value.numpy().flatten()

            def gradf(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                _, grad = self.neg_log_lik(locs, z, X, params, True)
                return grad.numpy()

            return fun, gradf
        else:
            def fun(params: np.ndarray):
                params = torch.tensor(params, dtype=torch.float64)
                value = self.neg_log_lik(locs, z, X, params, False)
                return value.numpy().flatten()
            return fun

    def get_local_pos(self, j: int, params_lik: torch.Tensor):
        # list of 1D params

        data = self.dis_data[j]
        # Extract the first two columns as local locations
        local_locs = data[:, :2]
        # Extract the third column as local z
        local_z = data[:, 2].reshape(-1, 1)
        if self.p > 0:
            # Extract columns from the fourth to the end as local X
            local_X = data[:, 3:]
        n = len(local_z)
        beta, delta, theta = self.vector2arguments_lik(params_lik)
        # Compute the kernel matrices
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.linalg.inv(K)
        B = self.kernel(local_locs, self.knots, theta) @ invK
        tempM = invK+delta*B.T@B
        if self.p > 0:
            errorv = local_X@beta-local_z
        else:
            errorv = -local_z
        Sigma = torch.linalg.inv(tempM/n)/n
        mu = Sigma@(delta*B.T@(-errorv))

        return (mu, Sigma)

    def get_pos(self, params_lik: torch.Tensor):

        beta, delta, theta = self.vector2arguments_lik(params_lik)
        local_locs_list = []
        local_z_list = []
        local_X_list = []
        N = 0

        # Iterate through the distributed data
        for data in self.dis_data:
            # Extract the first two columns as local locations
            local_locs = data[:, :2]
            # Extract the third column as local z
            local_z = data[:, 2].reshape(-1, 1)

            n = len(local_z)
            local_locs_list.append(local_locs)
            local_z_list.append(local_z)
            if self.p > 0:
                # Extract columns from the fourth to the end as local X
                local_X = data[:, 3:]
                local_X_list.append(local_X)
            N += n

        locs = torch.cat(local_locs_list, dim=0)
        z = torch.cat(local_z_list, dim=0)
        if self.p > 0:
            X = torch.cat(local_X_list, dim=0)
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.linalg.inv(K)
        B = self.kernel(locs, self.knots, theta) @ invK
        tempM = invK+delta*B.T@B
        if self.p > 0:
            errorv = X@beta-z
        else:
            errorv = -z
        Sigma = torch.linalg.inv(tempM/N)/N
        mu = Sigma@(delta*B.T@(-errorv))

        return (mu, Sigma)

    def get_local_minimizer(self, j: int, x0: torch.Tensor, thread_num=None):
        """
        Optimize the local likelihood functions in each machine to obtain the initial points
        """
        if thread_num != None:
            torch.set_num_threads(thread_num)
        x0 = x0.numpy()
        loc_nllikf, loc_nllikgf = self.local_neg_log_lik_wrapper(
            j, requires_grad=True)
        # options = {'maxiter': 100}
        # result = minimize(fun=loc_nllikf,
        #                   x0=x0,
        #                   method="CG",
        #                   jac=loc_nllikgf, options=options)
        # x0 = result.x
        result = minimize(fun=loc_nllikf,
                          x0=x0,
                          method="BFGS",
                          jac=loc_nllikgf)
        local_minimizer_lik = torch.tensor(result.x, dtype=torch.float64)
        mu, Sigma = self.get_local_pos(j, local_minimizer_lik)
        beta, delta, theta = self.vector2arguments_lik(local_minimizer_lik)
        # local_minimizer=self.argument2vector(mu,Sigma,beta,delta,theta)

        return (mu, Sigma, beta, delta, theta, result)

    def get_local_minimizers_parallel(self, x0, job_num=-1, thread_num=None):
        mu_list = []
        Sigma_list = []
        beta_list = []
        delta_list = []
        theta_list = []
        result_list = []
        except_list = []

        def compute_minimizer(j):
            try:
                mu, Sigma, beta, delta, theta, result = self.get_local_minimizer(
                    j, x0, thread_num)
                return mu, Sigma, beta, delta, theta, result, None  # No exception
            except Exception as e:
                return None, None, None, None, None, None, j  # Capture the exception

        # Parallelize over range(self.J) using joblib
        results = Parallel(n_jobs=job_num)(
            delayed(compute_minimizer)(j) for j in range(self.J))

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

    def get_local_minimizers(self, x0: torch.Tensor):
        mu_list = []
        Sigma_list = []
        beta_list = []
        delta_list = []
        theta_list = []
        result_list = []
        except_list = []
        for j in range(self.J):
            try:
                mu, Sigma, beta, delta, theta, result = self.get_local_minimizer(
                    j, x0)
                mu_list.append(mu)
                Sigma_list.append(Sigma)
                beta_list.append(beta)
                theta_list.append(theta)
                delta_list.append(delta)
                result_list.append(result)
            except Exception:
                except_list.append(j)
        return mu_list, Sigma_list, beta_list, delta_list, theta_list, result_list, except_list

    def get_minimier(self, x0: torch.Tensor, thread_num=None):
        if thread_num != None:
            torch.set_num_threads(thread_num)
        x0 = x0.numpy()
        nllikf, nllikgf = self.neg_log_lik_wrapper(requires_grad=True)
        result = minimize(fun=nllikf,
                          x0=x0,
                          method="BFGS",
                          jac=nllikgf, tol=1e-10)
        minimizer_lik = torch.tensor(result.x, dtype=torch.float64)
        mu, Sigma = self.get_pos(minimizer_lik)
        beta, delta, theta = self.vector2arguments_lik(minimizer_lik)

        return (mu, Sigma, beta, delta, theta, result)

    def de_optimize_stage2(self, mu_list: list[torch.Tensor], Sigma_list: list[torch.Tensor], beta_list: list[torch.Tensor], delta_list: list[torch.Tensor], theta_list: list[torch.Tensor], T: int, weights_round=4, seed=2024):
        torch.manual_seed(seed)
        # self.weights=torch.ones((self.J,self.J),dtype=torch.double)/self.J
        self.weights = torch.matrix_power(self.weights, weights_round)
        # define some functions

        def K_f(theta):
            return self.kernel(self.knots, self.knots, theta)

        def local_B_f(j, theta):
            data = self.dis_data[j]
            local_locs = data[:, :2]
            K = K_f(theta)
            invK = torch.linalg.inv(K)
            B = self.kernel(local_locs, self.knots, theta) @ invK
            return B

        def local_size_f(j):
            return self.dis_data[j].shape[0]

        def local_erorrV_f(j, beta):
            data = self.dis_data[j]
            # Extract the third column as local z
            local_z = data[:, 2].reshape(-1, 1)
            if self.p > 0:
                # Extract columns from the fourth to the end as local X
                local_X = data[:, 3:]
                errorv = local_X@beta-local_z
            else:
                errorv = -local_z
            return errorv

        def local_X_f(j):
            data = self.dis_data[j]
            local_X = data[:, 3:]
            return local_X

        def local_z_f(j):
            data = self.dis_data[j]
            local_z = data[:, 2].reshape(-1, 1)
            return local_z

        if self.p > 0:
            y_XTX_Mstack = torch.zeros(
                (self.p, self.p, self.J), dtype=torch.double)

            def compute_XTX_j(j):
                local_X = local_X_f(j)
                return local_X.T @ local_X
            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_XTX_j)(j)
                                          for j in range(self.J))
            # Stack the results
            for j in range(self.J):
                y_XTX_Mstack[:, :, j] = results[j]

        def y_mu_f_parallel(beta_list, theta_list):
            y_mu_Mstack = torch.zeros((self.m, self.J), dtype=torch.double)

            def compute_j(j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                return -local_B.T @ local_errorV

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(j,
                                                             beta_list[j], theta_list[j]) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_mu_Mstack[:, j:(j+1)] = results[j]

            return y_mu_Mstack

        def y_Sigma_f_parallel(theta_list):
            y_Sigma_Mstack = torch.zeros(
                (self.m, self.m, self.J), dtype=torch.double)

            def compute_j(j, theta_j):
                local_B = local_B_f(j, theta_j)
                return local_B.T @ local_B

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)
                                          (j, theta_list[j]) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_Sigma_Mstack[:, :, j] = results[j]

            return y_Sigma_Mstack

        if self.p > 0:
            def y_beta_f_parallel(mu_list, theta_list):
                y_beta_Mstack = torch.zeros(
                    (self.p, self.J), dtype=torch.double)

                def compute_j(j, mu_j, theta_j):
                    local_X = local_X_f(j)
                    local_B = local_B_f(j, theta_j)
                    local_z = local_z_f(j)
                    return local_X.T @ (local_z - local_B @ mu_j)

                # Parallel execution for each j
                results = Parallel(
                    n_jobs=-1)(delayed(compute_j)(j, mu_list[j], theta_list[j]) for j in range(self.J))

                # Stack the results
                for j in range(self.J):
                    y_beta_Mstack[:, j:(j+1)] = results[j]

                return y_beta_Mstack

        def y_delta_f_parallel(mu_list, Sigma_list, beta_list, theta_list):
            y_delta_Mstack = torch.zeros((1, self.J), dtype=torch.double)

            def compute_j(j, mu_j, Sigma_j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)

                term1 = torch.trace(local_B.T @ local_B @
                                    (Sigma_j + mu_j @ mu_j.T))
                term2 = 2 * local_errorV.T @ local_B @ mu_j
                term3 = local_errorV.T @ local_errorV

                return term1 + term2 + term3

            # Parallel execution for each j
            results = Parallel(n_jobs=-1)(delayed(compute_j)(
                j, mu_list[j], Sigma_list[j], beta_list[j], theta_list[j]) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_delta_Mstack[:, j] = results[j]

            return y_delta_Mstack

        def y_value_f(mu_list, Sigma_list, beta_list, delta_list, theta_list):
            y_value_M = torch.zeros((1, 1), dtype=torch.double)
            for j in range(self.J):
                local_value = self.local_value(
                    j, mu_list[j], Sigma_list[j], beta_list[j], delta_list[j], theta_list[j])
                y_value_M += local_value.reshape(-1, 1)
            y_value_M = y_value_M/self.J
            return y_value_M

        def com_value_f(mu_list, Sigma_list, theta_list):
            com_value_M = torch.zeros((1, 1), dtype=torch.double)
            for j in range(self.J):
                com_value = self.com_value(
                    mu_list[j], Sigma_list[j], theta_list[j])
                com_value_M += com_value.reshape(-1, 1)
            com_value_M = com_value_M/self.J
            return com_value_M

        def y_theta_f_parallel(mu_list, Sigma_list, beta_list, delta_list, theta_list):

            y_theta_Mstack = torch.zeros((2, self.J), dtype=torch.double)

            def compute_local_grad(j, mu, Sigma, beta, delta, theta):
                local_g_theta = self.local_grad_theta(
                    j, mu, Sigma, beta, delta, theta)
                return j, local_g_theta.reshape(-1, 1)

            # Parallelize the computation using joblib
            results = Parallel(n_jobs=-1)(delayed(compute_local_grad)(
                j, mu_list[j], Sigma_list[j], beta_list[j], delta_list[j], theta_list[j]) for j in range(self.J))

            # Populate y_theta_Mstack with the results
            for j, local_g_theta in results:
                y_theta_Mstack[:, j:(j + 1)] = local_g_theta

            # Optionally move results back to CPU
            return y_theta_Mstack  # Return as a CPU tensor if needed

        def y_hessian_theta_f_parallel(mu_list, Sigma_list, beta_list, delta_list, theta_list):
            hessian_theta_Mstack = torch.zeros(
                (2, 2, self.J), dtype=torch.double)

            def compute_local_hessian(j, mu, Sigma, beta, delta, theta):
                local_h_theta = self.local_hessian_theta(
                    j, mu, Sigma, beta, delta, theta)
                return j, local_h_theta
            results = Parallel(n_jobs=-1)(delayed(compute_local_hessian)(
                j, mu_list[j], Sigma_list[j], beta_list[j], delta_list[j], theta_list[j]) for j in range(self.J))

            for j, local_h_theta in results:
                hessian_theta_Mstack[:, :, j] = local_h_theta

            return hessian_theta_Mstack

        def com_grad_theta_f_parallel(mu_list, Sigma_list, theta_list):
            com_grad_theta_Mstack = torch.zeros(
                (2, self.J), dtype=torch.double)

            def compute_local_grad(j, mu, Sigma, theta):
                local_g_theta = self.com_grad_theta(mu, Sigma, theta)
                return j, local_g_theta.reshape(-1, 1)
            results = Parallel(n_jobs=-1)(delayed(compute_local_grad)(j,
                                                                      mu_list[j], Sigma_list[j], theta_list[j]) for j in range(self.J))

            for j, local_g_theta in results:
                com_grad_theta_Mstack[:, j:(j + 1)] = local_g_theta

            return com_grad_theta_Mstack

        def com_hessian_theta_f_parallel(mu_list, Sigma_list, theta_list):
            com_hessian_theta_Mstack = torch.zeros(
                (2, 2, self.J), dtype=torch.double)

            def compute_local_hessian(j, mu, Sigma, theta):
                com_h_theta = self.com_hessian_theta(mu, Sigma, theta)
                return j, com_h_theta
            results = Parallel(n_jobs=-1)(delayed(compute_local_hessian)(j,
                                                                         mu_list[j], Sigma_list[j], theta_list[j]) for j in range(self.J))

            for j, com_h_theta in results:
                com_hessian_theta_Mstack[:, :, j] = com_h_theta
            return com_hessian_theta_Mstack

        def list2Mstack(lt: list[torch.Tensor]):
            J = len(lt)

            if lt[0].dim() == 0:
                Mstack = torch.zeros((1, J), dtype=torch.double)
                for j in range(J):
                    Mstack[:, j] = lt[j]
            elif lt[0].dim() == 1:
                r = lt[0].shape[0]
                Mstack = torch.zeros((r, J), dtype=torch.double)
                for j in range(J):
                    Mstack[:, j] = lt[j]
            else:
                r = lt[0].shape[0]
                c = lt[0].shape[1]
                if c == 1:
                    Mstack = torch.zeros((r, J), dtype=torch.double)
                    for j in range(J):
                        Mstack[:, j] = lt[j].squeeze()
                else:
                    Mstack = torch.zeros((r, c, J), dtype=torch.double)
                    for j in range(J):
                        Mstack[:, :, j] = lt[j]

            return Mstack

        def Mstack2list(Mstack: torch.Tensor):
            lt = []
            J = Mstack.shape[-1]
            d = Mstack.dim()
            for j in range(J):
                if d == 2:
                    if Mstack.shape[0] == 1:
                        lt.append(Mstack[:, j])
                    else:
                        lt.append(Mstack[:, j].unsqueeze(1))
                else:
                    lt.append(Mstack[:, :, j])
            return lt

        def avg_local_minimizer(mu_list, Sigma_list, beta_list, delta_list, theta_list):
            mu_Mstack = list2Mstack(mu_list)
            Sigma_Mstack = list2Mstack(Sigma_list)
            beta_Mstack = list2Mstack(beta_list)
            delta_Mstack = list2Mstack(delta_list)
            theta_Mstack = list2Mstack(theta_list)
            mu_Mstack = torch.tensordot(mu_Mstack, self.weights, dims=1)
            Sigma_Mstack = torch.tensordot(Sigma_Mstack, self.weights, dims=1)
            beta_Mstack = torch.tensordot(beta_Mstack, self.weights, dims=1)
            delta_Mstack = torch.tensordot(delta_Mstack, self.weights, dims=1)
            theta_Mstack = torch.tensordot(theta_Mstack, self.weights, dims=1)

            mu_list = Mstack2list(mu_Mstack)
            Sigma_list = Mstack2list(Sigma_Mstack)
            beta_list = Mstack2list(beta_Mstack)
            delta_list = Mstack2list(delta_Mstack)
            theta_list = Mstack2list(theta_Mstack)
            return mu_list, Sigma_list, beta_list, delta_list, theta_list

        size_stack = torch.zeros((1, self.J), dtype=torch.double)
        for j in range(self.J):
            size_stack[:, j] = local_size_f(j)

        # mu_list,Sigma_list,beta_list,delta_list,theta_list=avg_local_minimizer(mu_list,Sigma_list,beta_list,delta_list,theta_list)

        # mu_lists=[mu_list]
        # Sigma_lists=[Sigma_list]
        beta_lists = [beta_list]
        delta_lists = [delta_list]
        theta_lists = [theta_list]

        for t in range(T):
            mu_list_p = mu_list
            Sigma_list_p = Sigma_list
            if t%10==0:
                print(f"iteration:{t}", end=', ')
            #print(f"iteration:{t}")
            # mu and Sigma
            if t == 0:
                y_mu_Mstack = torch.tensordot(y_mu_f_parallel(
                    beta_lists[0], theta_lists[0]), self.weights, dims=1)
                y_Sigma_Mstack = torch.tensordot(
                    y_Sigma_f_parallel(theta_lists[0]), self.weights, dims=1)
            else:
                y_mu_Mstack = torch.tensordot(y_mu_Mstack+y_mu_f_parallel(beta_lists[t], theta_lists[t])-y_mu_f_parallel(
                    beta_lists[t-1], theta_lists[t-1]), self.weights, dims=1)
                y_Sigma_Mstack = torch.tensordot(y_Sigma_Mstack+y_Sigma_f_parallel(
                    theta_lists[t])-y_Sigma_f_parallel(theta_lists[t-1]), self.weights, dims=1)

            def compute_mu_sigma(j, theta_j, delta_j):
                y_Sigma = y_Sigma_Mstack[:, :, j]
                y_Sigma = replace_negative_eigenvalues_with_zero(
                    y_Sigma)  # Ensure positive definiteness
                y_mu = y_mu_Mstack[:, j].unsqueeze(1)

                # Compute K and its inverse
                K = K_f(theta_j)
                invK = torch.linalg.inv(K)

                # Compute Sigma and mu
                Sigma = torch.linalg.inv(delta_j * self.J * y_Sigma + invK)
                mu = torch.linalg.inv(
                    delta_j * y_Sigma + invK / self.J) @ (delta_j * y_mu)

                return mu, Sigma

            # Parallel execution for all j
            results = Parallel(n_jobs=-1)(delayed(compute_mu_sigma)(j,
                                                                    theta_lists[t][j], delta_lists[t][j]) for j in range(self.J))

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
            if self.p > 0:
                y_XTX_Mstack = torch.tensordot(
                    y_XTX_Mstack, self.weights, dims=1)
                if t == 0:
                    y_beta_Mstack = torch.tensordot(y_beta_f_parallel(
                        mu_list, theta_lists[0]), self.weights, dims=1)
                else:
                    y_beta_Mstack = torch.tensordot(y_beta_Mstack+y_beta_f_parallel(
                        mu_list, theta_lists[t])-y_beta_f_parallel(mu_list_p, theta_lists[t-1]), self.weights, dims=1)
                beta_list = []
                for j in range(self.J):
                    y_XTX = y_XTX_Mstack[:, :, j]
                    y_beta = y_beta_Mstack[:, j].unsqueeze(1)
                    beta = torch.linalg.inv(y_XTX)@y_beta
                    beta_list.append(beta)
                beta_lists.append(beta_list)
            else:
                for j in range(self.J):
                    beta = None
                    beta_list.append(beta)
                beta_lists.append(beta_list)

            # delta
            if t == 0:
                y_delta_Mstack = torch.tensordot(y_delta_f_parallel(
                    mu_list, Sigma_list, beta_lists[1], theta_lists[0]), self.weights, dims=1)
            else:
                y_delta_Mstack = torch.tensordot(y_delta_Mstack+y_delta_f_parallel(mu_list, Sigma_list, beta_lists[t+1], theta_lists[t])-y_delta_f_parallel(
                    mu_list_p, Sigma_list_p, beta_lists[t], theta_lists[t-1]), self.weights, dims=1)

            size_stack = torch.tensordot(size_stack, self.weights, dims=1)

            delta_Mstack = size_stack/y_delta_Mstack
            # make sure it is positive
            delta_Mstack = torch.clamp(delta_Mstack, min=0)
            delta_list = Mstack2list(delta_Mstack)
            delta_lists.append(delta_list)

            # multi-round iteration for theta
            S = 6
            theta_list = theta_lists[t]
            s_list = []
            # weights=torch.ones((self.J,self.J),dtype=torch.double)/self.J
            weights = self.weights
            # weights=torch.matrix_power(self.weights, 5)
            for s in range(S):

                if t == 0 and s == 0:
                    y_theta_Mstack = torch.tensordot(y_theta_f_parallel(
                        mu_list, Sigma_list, beta_lists[1], delta_lists[1], theta_list), weights, dims=1)
                    y_hessian_theta_Mstack = torch.tensordot(y_hessian_theta_f_parallel(
                        mu_list, Sigma_list, beta_lists[1], delta_lists[1], theta_list), weights, dims=1)
                elif t >= 1 and s == 0:
                    y_theta_Mstack = torch.tensordot(y_theta_Mstack+y_theta_f_parallel(mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list)-y_theta_f_parallel(
                        mu_list_p, Sigma_list_p, beta_lists[t], delta_lists[t], theta_list_p), weights, dims=1)
                    y_hessian_theta_Mstack = torch.tensordot(y_hessian_theta_Mstack+y_hessian_theta_f_parallel(
                        mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list)-y_hessian_theta_f_parallel(mu_list_p, Sigma_list_p, beta_lists[t], delta_lists[t], theta_list_p), weights, dims=1)

                else:
                    y_theta_Mstack = torch.tensordot(y_theta_Mstack+y_theta_f_parallel(mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list)-y_theta_f_parallel(
                        mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list_p), weights, dims=1)
                    y_hessian_theta_Mstack = torch.tensordot(y_hessian_theta_Mstack+y_hessian_theta_f_parallel(
                        mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list)-y_hessian_theta_f_parallel(mu_list, Sigma_list, beta_lists[t+1], delta_lists[t+1], theta_list_p), weights, dims=1)

                # there are some modifications
                # com_grad_theta_Mstack=com_grad_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                # com_hessian_theta_Mstack=com_hessian_theta_f(mu_lists[t+1],Sigma_lists[t+1],theta_list)
                com_grad_theta_Mstack = torch.tensordot(com_grad_theta_f_parallel(
                    mu_list, Sigma_list, theta_list), weights, dims=1)
                com_hessian_theta_Mstack = torch.tensordot(com_hessian_theta_f_parallel(
                    mu_list, Sigma_list, theta_list), weights, dims=1)
                grad_theta_Mstack = y_theta_Mstack*self.J+com_grad_theta_Mstack

                # print(torch.norm(torch.mean(grad_theta_Mstack,dim=1)))
                if s >= 6 and torch.norm(torch.mean(grad_theta_Mstack, dim=1)) < 1e-4:
                    break
                hessian_theta_Mstack = y_hessian_theta_Mstack*self.J+com_hessian_theta_Mstack

                theta_Mstack = list2Mstack(theta_list)
                invh_m_grad_Mstack = torch.zeros_like(grad_theta_Mstack)

                noise = torch.randn(size=(2, 1))*0.01
                for j in range(self.J):
                    hess = hessian_theta_Mstack[:, :, j]
                    grad = grad_theta_Mstack[:, j].unsqueeze(1)

                    eigenvalues, eigenvectors = torch.linalg.eigh(hess)
                    # Take the absolute value of each eigenvalue
                    abs_eigenvalues = eigenvalues.abs()

                    # Define a positive threshold
                    threshold = 1e-20

                    # Replace elements smaller than the threshold with the threshold value
                    modified_eigenvalues = torch.where(
                        abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)

                    # Construct the diagonal matrix of modified eigenvalues
                    modified_eigenvalue_matrix = torch.diag(
                        modified_eigenvalues)

                    modified_hess = eigenvectors@modified_eigenvalue_matrix@eigenvectors.T
                    invh_m_grad_Mstack[:, j:(
                        j+1)] = torch.linalg.inv(modified_hess)@grad
                    if torch.all(eigenvalues > 0):
                        invh_m_grad_Mstack[:, j:(
                            j+1)] = torch.linalg.inv(modified_hess)@grad
                    else:
                        invh_m_grad_Mstack[:, j:(
                            j+1)] = 0.1*torch.linalg.inv(modified_hess)@grad

                # set the step size via backtracking line search
                if s >= 6 and torch.norm(torch.mean(invh_m_grad_Mstack, dim=1)) < 1e-5:
                    break
    #
                step_size = 1
                shrink_rate = 0.5
                Continue = True
            

                while Continue:
                    theta_Mstack_new = torch.tensordot(
                        theta_Mstack-step_size*invh_m_grad_Mstack, self.weights, dims=1)
                    theta_list_new = Mstack2list(theta_Mstack_new)
                    if torch.all(theta_Mstack_new > 0.005):

                        if s >= 1 and step_size > 0.0001 and torch.norm(torch.mean(grad_theta_Mstack, dim=1)) >= torch.norm(torch.mean(grad_theta_Mstack_p, dim=1)):
                            step_size *= shrink_rate
                        else:
                            theta_list_p = theta_list.copy()
                            theta_list = theta_list_new
                            Continue = False
                    else:
                        if step_size > 0.01:
                            step_size *= shrink_rate
                        else:
                            theta_Mstack = theta_Mstack+noise
                            Continue = False
               
                #print(f"theta:{torch.mean(theta_Mstack,dim=1).numpy()},gradient:{torch.mean(grad_theta_Mstack,dim=1).numpy()},norm of grad:{torch.norm(torch.mean(grad_theta_Mstack,dim=1)).numpy()}")
                
                if s >= 6 and torch.norm(torch.mean(grad_theta_Mstack, dim=1)-torch.mean(grad_theta_Mstack_p, dim=1)) < 1e-4:
                    break
                grad_theta_Mstack_p = grad_theta_Mstack
            theta_lists.append(theta_list)
            s_list.append(s)

        f_value = y_value_f(mu_list, Sigma_list, beta_lists[T], delta_lists[T],
                            theta_lists[T])*self.J+com_value_f(mu_list, Sigma_list, theta_lists[T])
        print(f_value)

        return mu_list, Sigma_list, beta_lists, delta_lists, theta_lists, s_list, f_value

    def ce_optimize_stage2(self, mu: torch.Tensor, Sigma: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor, T: int, job_num, seed=2024, thread_num=None, backend='threading'):
        torch.manual_seed(seed)
        if thread_num != None:
            torch.set_num_threads(thread_num)
        # define some functions

        def K_f(theta):
            return self.kernel(self.knots, self.knots, theta)

        def local_B_f(j, theta):
            data = self.dis_data[j]
            local_locs = data[:, :2]
            K = K_f(theta)
            invK = torch.linalg.inv(K)
            B = self.kernel(local_locs, self.knots, theta) @ invK
            return B

        def local_size_f(j):
            return self.dis_data[j].shape[0]

        def local_erorrV_f(j, beta):
            data = self.dis_data[j]
            # Extract the third column as local z
            local_z = data[:, 2].reshape(-1, 1)
            # Extract columns from the fourth to the end as local X
            local_X = data[:, 3:]
            errorv = local_X@beta-local_z
            return errorv

        def local_X_f(j):
            data = self.dis_data[j]
            local_X = data[:, 3:]
            return local_X

        def local_z_f(j):
            data = self.dis_data[j]
            local_z = data[:, 2].reshape(-1, 1)
            return local_z

        # parallel
        y_XTX = torch.zeros((self.p, self.p), dtype=torch.double)

        def compute_XTX_j(j):
            local_X = local_X_f(j)
            return local_X.T @ local_X
        results = Parallel(n_jobs=job_num, backend=backend)(
            delayed(compute_XTX_j)(j) for j in range(self.J))
        for j in range(self.J):
            y_XTX += results[j]
        y_XTX = y_XTX/self.J

    
        def y_mu_f_parallel(beta, theta):
            y_mu_M = torch.zeros((self.m, 1), dtype=torch.double)

            def compute_j(j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)
                return -local_B.T @ local_errorV

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j, beta, theta) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_mu_M = y_mu_M + results[j]
            y_mu_M = y_mu_M/self.J
            return y_mu_M


        def y_Sigma_f_parallel(theta):
            y_Sigma_M = torch.zeros((self.m, self.m), dtype=torch.double)

            def compute_j(j, theta_j):
                local_B = local_B_f(j, theta_j)
                return local_B.T @ local_B

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j, theta) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_Sigma_M += results[j]
            y_Sigma_M = y_Sigma_M/self.J
            return y_Sigma_M


        def y_beta_f_parallel(mu, theta):
            y_beta_M = torch.zeros((self.p, 1), dtype=torch.double)

            def compute_j(j, mu_j, theta_j):
                local_X = local_X_f(j)
                local_B = local_B_f(j, theta_j)
                local_z = local_z_f(j)
                return local_X.T @ (local_z - local_B @ mu_j)

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j, mu, theta) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_beta_M += results[j]
            y_beta_M = y_beta_M/self.J
            return y_beta_M


        def y_delta_f_parallel(mu, Sigma, beta, theta):
            y_delta = torch.zeros((1, 1), dtype=torch.double)

            def compute_j(j, mu_j, Sigma_j, beta_j, theta_j):
                local_B = local_B_f(j, theta_j)
                local_errorV = local_erorrV_f(j, beta_j)

                term1 = torch.trace(local_B.T @ local_B @
                                    (Sigma_j + mu_j @ mu_j.T))
                term2 = 2 * local_errorV.T @ local_B @ mu_j
                term3 = local_errorV.T @ local_errorV

                return term1 + term2 + term3

            # Parallel execution for each j
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j, mu, Sigma, beta, theta) for j in range(self.J))

            # Stack the results
            for j in range(self.J):
                y_delta += results[j]
            y_delta = y_delta/self.J
            return y_delta

        def y_theta_f_parallel(mu, Sigma, beta, delta, theta):

            y_theta_M = torch.zeros((2, 1), dtype=torch.double)

            knots = self.knots
            # Compute the kernel matrices
            K = self.kernel(knots, knots, theta)
            invK = torch.linalg.inv(K)

            def compute_local_grad(j):
                data = self.dis_data[j]
                # Extract the first two columns as local locations
                local_locs = data[:, :2]
                # Extract the third column as local z
                local_z = data[:, 2].unsqueeze(1)
                local_X = data[:, 3:]

                n = local_z.shape[0]

                theta = theta.clone()
                if theta.grad is not None:
                    theta.grad.data.zero_()
                theta.requires_grad_(True)
                B = self.kernel(local_locs, knots, theta) @ invK
                grad = theta.grad
                # Compute the value of the local objective function
                errorV = local_X @ beta-local_z
                f_value = -n * torch.log(delta) + delta * (
                    torch.trace(B.T @ B @ (Sigma + mu @ mu.T))
                    + 2 * errorV.T @ B @ mu
                    + errorV.T @ errorV
                )
                f_value.backward()
                return grad

            def compute_local_grad(j, mu_j, Sigma_j, beta_j, delta_j, theta_j):
                local_g_theta = self.local_grad_theta(
                    j, mu_j, Sigma_j, beta_j, delta_j, theta_j)
                return j, local_g_theta.reshape(-1, 1)

            # Parallelize the computation using joblib
            results = Parallel(n_jobs=job_num, backend=backend)(delayed(
                compute_local_grad)(j, mu, Sigma, beta, delta, theta) for j in range(self.J))

            # Populate y_theta_Mstack with the results
            for j, local_g_theta in results:
                y_theta_M += local_g_theta
            y_theta_M = y_theta_M/self.J
            return y_theta_M


        def y_hessian_theta_f_parallel(mu, Sigma, beta, delta, theta):
            hessian_theta_M = torch.zeros((2, 2), dtype=torch.double)

            def compute_local_hessian(j, mu_j, Sigma_j, beta_j, delta_j, theta_j):
                local_h_theta = self.local_hessian_theta(
                    j, mu_j, Sigma_j, beta_j, delta_j, theta_j)
                return j, local_h_theta
            results = Parallel(n_jobs=job_num, backend=backend)(delayed(
                compute_local_hessian)(j, mu, Sigma, beta, delta, theta) for j in range(self.J))

            for j, local_h_theta in results:
                hessian_theta_M += local_h_theta
            hessian_theta_M = hessian_theta_M/self.J
            return hessian_theta_M

        def com_grad_theta_f(mu, Sigma, theta):
            com_grad_theta_M = torch.zeros((2, 1), dtype=torch.double)
            com_g_theta = self.com_grad_theta(mu, Sigma, theta)
            com_grad_theta_M = com_g_theta.reshape(-1, 1)
            return com_grad_theta_M

        def com_hessian_theta_f(mu, Sigma, theta):
            com_hessian_theta_M = torch.zeros((2, 2), dtype=torch.double)
            com_h_theta = self.com_hessian_theta(mu, Sigma, theta)
            com_hessian_theta_M = com_h_theta
            return com_hessian_theta_M

        size = torch.zeros((1, 1), dtype=torch.double)
        for j in range(self.J):
            size += local_size_f(j)
        size = size/self.J

        beta_list = [beta]
        delta_list = [delta]
        theta_list = [theta]
        s_list = []
        for t in range(T):
            # if t % 10 == 0:
            #     print(f"iteration:{t}", end=', ')
            print(f"iteration:{t}")
            # mu and Sigma
            y_mu = y_mu_f_parallel(beta_list[t], theta_list[t])
            y_Sigma = y_Sigma_f_parallel(theta_list[t])
            K = K_f(theta_list[t])
            invK = torch.linalg.inv(K)
            Sigma = torch.linalg.inv(delta_list[t]*self.J*y_Sigma+invK)
            mu = torch.linalg.inv(
                delta_list[t]*y_Sigma+invK/self.J)*delta_list[t]@y_mu

            # beta
            y_beta = y_beta_f_parallel(mu, theta_list[t])
            beta = torch.linalg.inv(y_XTX)@y_beta
            beta_list.append(beta)

            # delta
            y_delta = y_delta_f_parallel(
                mu, Sigma, beta_list[t+1], theta_list[t])
            delta = size/y_delta
            delta_list.append(delta)

            # multi-round iteration
            S = 5
            theta = theta_list[t]
            for s in range(S):
                y_theta = y_theta_f_parallel(
                    mu, Sigma, beta_list[t+1], delta_list[t+1], theta)
                y_hessian_theta = y_hessian_theta_f_parallel(
                    mu, Sigma, beta_list[t+1], delta_list[t+1], theta)

                com_grad_theta = com_grad_theta_f(mu, Sigma, theta)
                com_hessian_theta = com_hessian_theta_f(mu, Sigma, theta)
                grad = y_theta*self.J+com_grad_theta

                if torch.norm(grad) < 1e-4:
                    break

                hess = y_hessian_theta*self.J+com_hessian_theta
                eigenvalues, eigenvectors = torch.linalg.eigh(hess)

                # Take the absolute value of each eigenvalue
                abs_eigenvalues = eigenvalues.abs()

                # Define a positive threshold
                threshold = 0.01

                # Replace elements smaller than the threshold with the threshold value
                modified_eigenvalues = torch.where(
                    abs_eigenvalues < threshold, torch.tensor(threshold), abs_eigenvalues)

                # Construct the diagonal matrix of modified eigenvalues
                modified_eigenvalue_matrix = torch.diag(modified_eigenvalues)

                modified_hess = eigenvectors@modified_eigenvalue_matrix@eigenvectors.T


                invh_m_grad = torch.linalg.inv(modified_hess)@grad
                if torch.norm(invh_m_grad) < 1e-5:
                    break
                # set the step size via backtracking line search
                step_size = 0.4
                theta = theta-step_size*invh_m_grad
                # step_size = 1
                # shrink_rate = 0.8
                # m = 0.1
                # Continue = True
                # f_value = y_value_f_parallel(
                #     mu, Sigma, beta_list[t+1], delta_list[t+1], theta)*self.J+com_value_f(mu, Sigma, theta)

                # while Continue:
                #     theta_new = theta-step_size*invh_m_grad
                #     if torch.all(theta_new > 0.005):
                #         f_value_new = y_value_f_parallel(
                #             mu, Sigma, beta_list[t+1], delta_list[t+1], theta_new)*self.J+com_value_f(mu, Sigma, theta_new)
                #         if f_value_new > f_value-m*step_size*torch.dot(grad.squeeze(), invh_m_grad.squeeze()):
                #             step_size *= shrink_rate
                #         else:
                #             theta = theta_new
                #             Continue = False
                #     else:
                #         if step_size > 1e-4:
                #             step_size *= shrink_rate
                #         else:
                #             theta = theta+noise
                #             Continue = False

                print(f"theta:{theta},gradient:{torch.norm(grad)}")
            theta_list.append(theta)
            s_list.append(s)

        return mu, Sigma, beta_list, delta_list, theta_list, s_list

    def ce_asy_variance(self, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor, job_num, seed=2024, thread_num=None, backend='threading',method='autodif'):
        
        if method=="autodif":
            return self.ce_asy_variance_autodif(beta,delta,theta,job_num,seed,thread_num,backend)
        else:
            return self.ce_asy_variance_explicit(beta,delta,theta,job_num,seed,thread_num,backend)
    def ce_asy_variance_autodif(self, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor, job_num, seed=2024, thread_num=None, backend='threading'):
        # for beta
        torch.manual_seed(seed)
        if thread_num != None:
            torch.set_num_threads(thread_num)
        if self.p>0:
            theta = theta.clone().detach().requires_grad_(True)
            K = self.kernel(self.knots, self.knots, theta)
            invK = torch.linalg.inv(K)

            def compute_j(j):
                local_data = self.dis_data[j]
                local_locs = local_data[:, :2]
                local_z = local_data[:, 2].reshape(-1, 1)
                local_X = local_data[:, 3:]
                local_size = local_data.shape[0]
                local_errorv = (local_X@beta).reshape(-1,1)-local_z
                local_B = self.kernel(local_locs, self.knots, theta) @ invK
                local_XTX = local_X.T @ local_X
                local_BTB = local_B.T @ local_B
                local_XTB = local_X.T @ local_B
                local_errorvTB = local_errorv.T@local_B
                local_errorvsquare = (local_errorv.T@local_errorv).item()
                return (local_XTX, local_BTB, local_XTB, local_size, local_errorvTB, local_errorvsquare)
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j) for j in range(self.J))
            XTX = torch.zeros((self.p, self.p), dtype=torch.double)
            BTB = torch.zeros((self.m, self.m), dtype=torch.double)
            XTB = torch.zeros((self.p, self.m), dtype=torch.double)
            size = 0
            errorvTB = torch.zeros((1, self.m), dtype=torch.double)
            errorvsquare = 0
            for j in range(self.J):
                result_j = results[j]
                XTX += result_j[0]
                BTB += result_j[1]
                XTB += result_j[2]
                size += result_j[3]
                errorvTB += result_j[4]
                errorvsquare += result_j[5]
            A = invK+delta*BTB
            invA = torch.linalg.inv(A)
            V_beta = (delta*XTX-(delta**2)*XTB@invA@XTB.T)/size
            V_delta_1 = (delta**2)*size
            V_delta_2 = -2*(delta**3)*torch.trace(invA@BTB)
            V_delta_3 = (delta**4)*torch.trace(invA@BTB@invA@BTB)
            V_delta = (V_delta_1+V_delta_2+V_delta_3)/(2*(delta**4)*size)
            logd = -size*torch.log(delta)+torch.logdet(K)+torch.logdet(A)
            quad = errorvsquare-(delta**2)*errorvTB@invA@errorvTB.T
            neglik = (logd+quad)/2

            gradients = grad(neglik, theta, create_graph=True)[0]  # First derivative
            hessian = torch.zeros((len(theta), len(theta)),dtype=torch.double)  # Initialize Hessian

            for i in range(len(theta)):
                second_grad = grad(gradients[i], theta, retain_graph=True)[0]  # Second derivative
                hessian[i, :] = second_grad
            V_theta = hessian/self.m

            return (V_beta, V_delta, V_theta)
        else:
            theta = theta.clone().detach().requires_grad_(True)
            K = self.kernel(self.knots, self.knots, theta)
            invK = torch.linalg.inv(K)

            def compute_j(j):
                local_data = self.dis_data[j]
                local_locs = local_data[:, :2]
                local_z = local_data[:, 2].reshape(-1, 1)
                local_size = local_data.shape[0]
                local_errorv = -local_z
                local_B = self.kernel(local_locs, self.knots, theta) @ invK
                local_BTB = local_B.T @ local_B
                local_errorvTB = local_errorv.T@local_B
                local_errorvsquare = local_errorv.T@local_errorv
                return (local_BTB, local_size, local_errorvTB, local_errorvsquare)
            results = Parallel(n_jobs=job_num, backend=backend)(
                delayed(compute_j)(j) for j in range(self.J))
            BTB = torch.zeros((self.m, self.m), dtype=torch.double)
            size = 0
            errorvTB = torch.zeros((1, self.m), dtype=torch.double)
            errorvsquare = 0
            for j in range(self.J):
                result_j = results[j]
                BTB += result_j[0]
                size += result_j[1]
                errorvTB += result_j[2]
                errorvsquare += result_j[3]
            A = invK+delta*BTB
            invA = torch.linalg.inv(A)
            V_delta_1 = (delta**2)*size
            V_delta_2 = -2*(delta**3)*torch.trace(invA@BTB)
            V_delta_3 = (delta**4)*torch.trace(invA@BTB@invA@BTB)
            V_delta = (V_delta_1+V_delta_2+V_delta_3)/(2*(delta**4)*size)
            logd = -size*torch.log(delta)+torch.logdet(K)+torch.logdet(A)
            quad = errorvsquare-(delta**2)*errorvTB@invA@errorvTB.T
            neglik = (logd+quad)/2

            gradients = grad(neglik, theta, create_graph=True)[0]  # First derivative
            hessian = torch.zeros((len(theta), len(theta)),dtype=torch.double)  # Initialize Hessian

            for i in range(len(theta)):
                second_grad = grad(gradients[i], theta, retain_graph=True)[0]  # Second derivative
                hessian[i, :] = second_grad
            V_theta = hessian/self.m

            return (V_delta, V_theta)
            
    def ce_asy_variance_explicit(self, beta: torch.Tensor, delta: torch.Tensor, theta: torch.Tensor, job_num, seed=2024, thread_num=None, backend='threading'):
        #uncomplete
        torch.manual_seed(seed)
        if thread_num != None:
            torch.set_num_threads(thread_num)
        theta = theta.clone().detach().requires_grad_(True)
        K = self.kernel(self.knots, self.knots, theta)
        gradient = []
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                grad_element = torch.autograd.grad(K[i, j], theta, retain_graph=True)[0]
                gradient.append(grad_element)
        Kg = torch.stack(gradient).view(K.shape + theta.shape) #Row-major order
        
        invK = torch.linalg.inv(K)

        def compute_j(j):
            local_data = self.dis_data[j]
            local_locs = local_data[:, :2]
            local_z = local_data[:, 2].reshape(-1, 1)
            local_X = local_data[:, 3:]
            local_size = local_data.shape[0]
            local_errorv = local_X@beta-local_z
            local_B = self.kernel(local_locs, self.knots, theta) @ invK
            gradient = []
            for i in range(local_B.shape[0]):
                for l in range(local_B.shape[1]):
                    grad_element = torch.autograd.grad(local_B[i, l], theta, retain_graph=True)[0]
                    gradient.append(grad_element)
            local_Bg=torch.stack(gradient).view(local_B.shape + theta.shape)
            local_XTX = local_X.T @ local_X
            local_BTB = local_B.T @ local_B
            local_XTB = local_X.T @ local_B
            local_BTBg= torch.zeros(self.m,self.m,len(theta))
            local_BgTBg= torch.zeros(self.m,self.m,len(theta))
            for l in range(len(theta)):
                local_BTBg[:,:,l]=local_B.T @ local_Bg[:,:,l]
                local_BgTBg[:,:,l]=local_Bg.T @ local_Bg[:,:,l]
            return (local_XTX, local_BTB, local_XTB, local_size, local_BTBg, local_BgTBg)
        results = Parallel(n_jobs=job_num, backend=backend)(
            delayed(compute_j)(j) for j in range(self.J))
        XTX = torch.zeros((self.p, self.p), dtype=torch.double)
        BTB = torch.zeros((self.m, self.m), dtype=torch.double)
        XTB = torch.zeros((self.p, self.m), dtype=torch.double)
        size = 0
        BTBg = torch.zeros((self.m, self.m,self.p), dtype=torch.double)
        BgTBg = torch.zeros((self.m, self.m,self.p), dtype=torch.double)
        for j in range(self.J):
            result_j = results[j]
            XTX += result_j[0]
            BTB += result_j[1]
            XTB += result_j[2]
            size += result_j[3]
            BTBg += result_j[4]
            BgTBg += result_j[5]
        A = invK+delta*BTB
        invA = torch.linalg.inv(A)
        V_beta = (delta*XTX-(delta**2)*XTB@invA@XTB.T)/size
        V_delta_1 = (delta**2)*size
        V_delta_2 = -2*(delta**3)*torch.trace(invA@BTB)
        V_delta_3 = (delta**4)*torch.trace(invA@BTB@invA@BTB)
        V_delta = (V_delta_1+V_delta_2+V_delta_3)/(2*(delta**4)*size)
        V_theta_diag=torch.zeros((self.p,1))
        for l in range(self.p):
            BTBgl=BTBg[:,:,l]
            BgTBgl=BgTBg[:,:,l]
            Kgl=Kg[:,:,l]
            term1=torch.trace(K@BTBgl@K@BTBgl)+torch.trace(Kgl@BTB@Kgl@BTB)+torch.trace(Kgl@BTB@Kgl@BTB)
            

        return (V_beta, V_delta, V_theta_diag)
    
    def Hessian_delta_theta(self, delta_theta: torch.Tensor, job_num, seed=2024, thread_num=None, backend='threading'):
        #compute the Hessian of the log-likelihood function with respect to delta and theta
        torch.manual_seed(seed)
        if thread_num != None:
            torch.set_num_threads(thread_num)
        delta_theta = delta_theta.clone().detach().requires_grad_(True)
        delta = delta_theta[0]
        theta = delta_theta[1:]
        K = self.kernel(self.knots, self.knots, theta)
        invK = torch.linalg.inv(K)
        def compute_j(j):
                local_data: torch.Tensor = self.dis_data[j]
                local_locs = local_data[:, :2]
                local_z = local_data[:, 2].reshape(-1, 1)
                local_size = local_data.shape[0]
                local_errorv = -local_z
                local_B: torch.Tensor = self.kernel(local_locs, self.knots, theta) @ invK
                local_BTB = local_B.T @ local_B
                local_errorvTB = local_errorv.T@local_B
                local_errorvsquare = local_errorv.T@local_errorv
                return (local_BTB, local_size, local_errorvTB, local_errorvsquare)
        results = Parallel(n_jobs=job_num, backend=backend)(
            delayed(compute_j)(j) for j in range(self.J))
        BTB = torch.zeros((self.m, self.m), dtype=torch.double)
        size = 0
        errorvTB = torch.zeros((1, self.m), dtype=torch.double)
        errorvsquare = 0
        for j in range(self.J):
            result_j = results[j]
            BTB += result_j[0]
            size += result_j[1]
            errorvTB += result_j[2]
            errorvsquare += result_j[3]
        A = invK+delta*BTB
        invA = torch.linalg.inv(A)
        logd = -size*torch.log(delta)+torch.logdet(K)+torch.logdet(A)
        quad = errorvsquare-(delta**2)*errorvTB@invA@errorvTB.T 
        neglik = (logd+quad)/2
        gradients = grad(neglik, delta_theta, create_graph=True)[0]  # First derivative
        hessian = torch.zeros((len(delta_theta), len(delta_theta)),dtype=torch.double)  # Initialize Hessian
        for i in range(len(delta_theta)):
            second_grad = grad(gradients[i], delta_theta, retain_graph=True)[0]  # Second derivative
            hessian[i, :] = second_grad
        return hessian
    
    def Hessian_delta_theta_Expected(self, delta_theta: torch.Tensor,delta_theta_true: torch.Tensor):
        #compute the Hessian of the log-likelihood function with respect to delta and theta
    
        local_locs_list = []
        for data in self.dis_data:
            # Extract the first two columns as local locations
            local_locs = data[:, :2]
            # Extract the third column as local z
            local_locs_list.append(local_locs)
        locs = torch.cat(local_locs_list, dim=0)
        
        
        knots = self.knots
        N = locs.shape[0]
        
        delta_theta = delta_theta.clone().detach().requires_grad_(True)
        delta = delta_theta[0]
        theta = delta_theta[1:]
        
        
        


        # Compute the kernel matrices
        K = self.kernel(knots, knots, theta)
        invK = torch.linalg.inv(K)
        B = self.kernel(locs, knots, theta) @ invK

        tempM = invK+delta*B.T@B
        
        theta_true = delta_theta_true[1:]
        delta_true = delta_theta_true[0]
        K_true = self.kernel(knots, knots, theta_true)
        invK_true = torch.linalg.inv(K_true)
        B_true: torch.Tensor = self.kernel(locs, knots, theta_true) @ invK_true
        
        
        f_value = torch.logdet(tempM)+torch.logdet(K)-N*torch.log(delta)+N*delta/delta_true+delta*torch.trace(K_true@B_true.T@B_true)-delta**2/delta_true*torch.trace(torch.linalg.inv(tempM)@B.T@B)-delta**2*torch.trace(K_true@B_true.T@B@torch.linalg.inv(tempM)@B.T@B_true)
        

        gradients = grad(f_value, delta_theta, create_graph=True)[0]  # First derivative
        hessian = torch.zeros((len(delta_theta), len(delta_theta)),dtype=torch.double)  # Initialize Hessian
        for i in range(len(delta_theta)):
            second_grad = grad(gradients[i], delta_theta, retain_graph=True)[0]  # Second derivative
            hessian[i, :] = second_grad
        return hessian/2