import numpy as np
import cvxpy as cp


def metropolis_hastings_weight_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    W = np.zeros_like(adj_matrix, dtype=float)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                W[i, j] = 1 / (max(np.sum(adj_matrix[i]), np.sum(adj_matrix[j])) + 1)
        W[i, i] = 1 - np.sum(W[i])
    return W

def laplacian_matrix(adj_matrix):
    D = np.diag(np.sum(adj_matrix, axis=1))
    L = D - adj_matrix
    return L

def normalized_laplacian_matrix(adj_matrix):
    D = np.diag(np.sum(adj_matrix, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(adj_matrix, axis=1)))
    L = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return L

def averaging_weight_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    W = np.zeros_like(adj_matrix, dtype=float)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1 or i == j:
                W[i, j] = 1 / (np.sum(adj_matrix[i]) + 1)
    return W

def consensus_weight_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    W = np.zeros_like(adj_matrix, dtype=float)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                W[i, j] = 1 / np.sum(adj_matrix[i])
        W[i, i] = 1 - np.sum(W[i])
    return W

def maximum_degree_weight_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    W = np.zeros_like(adj_matrix, dtype=float)
    max_degree = np.max(np.sum(adj_matrix, axis=1))
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                W[i, j] = 1 / max_degree
        W[i, i] = 1 - np.sum(W[i])
    return W

def optimal_weight_matrix(adj_matrix):
    n = adj_matrix.shape[0]
    
    # Variables
    W = cp.Variable((n, n), symmetric=True)
    obj=cp.norm(W- (1/n) * np.ones((n, n)))
    # Constraints
    constraints = [
        W @ np.ones(n) == np.ones(n),  # W1 = 1
        W == W.T,  # W is symmetric
        W[adj_matrix == 0] == 0,  # W[i,j] = 0 if adj_matrix[i,j] = 0
    ]
    
    # Objective
    objective = cp.Minimize(obj)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    weight=W.value
    weight[adj_matrix == 0] = 0
    return weight, obj.value

