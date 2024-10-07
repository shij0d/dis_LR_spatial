import numpy as np
import torch
def create_matrix_with_one(m, n, i, j):
    """
    Creates an m x n matrix with a 1 at position (i, j) and 0s elsewhere.

    Parameters:
    m (int): Number of rows.
    n (int): Number of columns.
    i (int): Row index where the 1 should be placed.
    j (int): Column index where the 1 should be placed.

    Returns:
    np.ndarray: The resulting matrix.
    """
    matrix = np.zeros((m, n))
    matrix[i, j] = 1
    return matrix

def create_matrix_with_one_torch(m, n, i, j):
    matrix = torch.zeros((m, n))
    matrix[i, j] = 1
    return matrix

def softplus(x: np.ndarray) -> np.ndarray:
    """
    Compute the softplus function for the input array.
    
    The softplus function is a smooth approximation to the ReLU function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Output array where the softplus function has been applied element-wise.
    """
    return np.logaddexp(0, x)  # np.logaddexp(0, x) computes log(1 + exp(x)), which is the softplus function.

def softplus_torch(x: torch.Tensor) -> torch.Tensor:
    value=torch.logaddexp(torch.tensor(0), x)
    return value 

def inv_softplus_torch(x: torch.Tensor) -> torch.Tensor:
    # each element of x is required to be positive

    value=x+torch.log(1-torch.exp(-x))
    return value
def softplus_d(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the softplus function for the input array.
    
    The derivative of the softplus function is the sigmoid function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Output array where the derivative of the softplus function has been applied element-wise.
    """
    if not np.isscalar(x):
        x = np.array(x)  # Ensure x is an array for element-wise operations.

    absx = np.abs(x)  # Compute the absolute value of x.
    exp_absx = np.exp(-absx)  # Compute exp(-|x|) for the sigmoid calculation.

    # Compute the derivative of the softplus function element-wise.
    value = np.where(x > 0, 1 / (exp_absx + 1), exp_absx / (1 + exp_absx))

    if np.isscalar(x):
        value = value.item()  # Convert the single-element array back to a scalar if the input was a scalar.

    return value



def replace_negative_eigenvalues_with_zero(matrix:torch.Tensor):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Replace negative eigenvalues with zero
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    # Reconstruct the matrix with the modified eigenvalues
    matrix_modified = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    
    return matrix_modified