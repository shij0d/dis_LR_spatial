# Decentralized Inference for Spatial Data Under Low-Rank Models

A Python implementation of decentralized algorithms for parameter estimation and prediction under spatial low-rank models. See [the paper](https://arxiv.org/abs/2502.00309) for the background and the algorithm details.


## Installation

```bash
#create the working dir
mkdir Dis_Spatial
cd Dis_Spatial

# Clone the repository
git clone https://github.com/shij0d/dis_LR_spatial.git

# Install the package
pip install -e .
```

### Dependencies

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- NetworkX
- Matplotlib
- Joblib

## Quick Start

### 1. Import required packages
```python
import torch
from src.estimation_torch import GPPEstimation
from src.kernel import exponential_kernel
from src.networks import generate_connected_erdos_renyi_graph
from src.weights import optimal_weight_matrix
import networkx as nx
import numpy as np
```

### 2. Prepare Distributed Data

Data should be structured as a list of local datasets, where each local dataset is a PyTorch tensor with columns `[coord_x, coord_y, y, X_1, X_2, ..., X_p]`:

```python
# dis_data: list of tensors for J machines
# Each tensor has shape (n_j, 3 + p) where:
#   - columns 0-1: spatial coordinates (x, y)
#   - column 2: response variable y
#   - columns 3+: covariates X (optional)
dis_data = [local_data_1, local_data_2, ..., local_data_J]
```

### 3. Set Up the Network

Define the communication network topology and set/compute optimal weights, e.g.,
```python
J = 10  # Number of machines
con_pro = 0.5  # Connection probability

# Generate a connected Erdős-Rényi graph
er = generate_connected_erdos_renyi_graph(J, con_pro)
adj_matrix = nx.adjacency_matrix(er).todense()
np.fill_diagonal(adj_matrix, 1)

# Compute optimal weight matrix for consensus
weights, _ = optimal_weight_matrix(adj_matrix=adj_matrix)
weights = torch.tensor(weights, dtype=torch.float64)
```

### 4. Initialize the Estimator

```python
gpp_estimation = GPPEstimation(
    dis_data=dis_data,
    kernel=exponential_kernel,  # Kernel function
    knots=knots,                # Knot locations (m × 2 tensor), which can be grid points, or randomly selected over all locations (if the locations are available)
    weights=weights             # Network weight matrix (J × J)
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `dis_data` | `list[torch.Tensor]` | List of local datasets for each machine |
| `kernel` | `callable` | Positive-definite kernel function |
| `knots` | `torch.Tensor` | Knot locations for low-rank approximation (m × d) |
| `weights` | `torch.Tensor` | Network weight matrix for consensus (J × J) |

### 5. Compute Estimators Based on Local Data (Initialization)

Compute initial estimates on each machine in parallel:

```python
# x_true: initial parameter vector for optimization
mu_list, Sigma_list, beta_list, delta_list, theta_list, _, _ = \
    gpp_estimation.get_local_minimizers_parallel(x_true, job_num=-1)
```

> **Note:** Local estimators may vary significantly across machines. For robustness, apply a decentralized aggregation method (e.g., averaging or median) before the main optimization.

### 6. Decentralized Parameter Estimation

Run the decentralized optimization algorithm:

```python
T = 100  # Number of iterations

de_estimators = gpp_estimation.de_optimize_stage2(
    mu_list, Sigma_list, beta_list, delta_list, theta_list,
    T=T,
    weights_round=6  # Rounds of weight matrix refinement
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `mu_list` | `list[torch.Tensor]` | Initial mean estimates for each machine |
| `Sigma_list` | `list[torch.Tensor]` | Initial covariance estimates |
| `beta_list` | `list[torch.Tensor]` | Initial regression coefficients |
| `delta_list` | `list[torch.Tensor]` | Initial noise precision estimates |
| `theta_list` | `list[torch.Tensor]` | Initial kernel parameters |
| `T` | `int` | Number of optimization iterations |
| `weights_round` | `int` | Number of rounds to improve network connectivity |

### 7. Prediction

Use the estimated parameters for spatial prediction in each machine:

```python
from src.prediction import GPPPrediction

# Extract final estimates (e.g., from machine 0)
mu_final = de_estimators['mu'][0]
Sigma_final = de_estimators['Sigma'][0]
beta_final = de_estimators['beta'][0]
delta_final = de_estimators['delta'][0]
theta_final = de_estimators['theta'][0]

# Initialize predictor
predictor = GPPPrediction(
    location=new_locations,    # Prediction locations (N_pred × 2)
    kernel=exponential_kernel,
    knots=knots,
    X=X_new,                   # Covariates at new locations (optional)
    mu=mu_final,
    Sigma=Sigma_final,
    beta=beta_final,
    delta=delta_final,
    theta=theta_final
)

# Get predictions
mean_pred, cov_pred = predictor.predict()
```

## Available Kernels

The package provides several kernel functions in `src/kernel.py`:

| Kernel | Function | Description |
|--------|----------|-------------|
| Exponential | `exponential_kernel` | Matérn kernel with ν = 0.5 |
| Matérn (ν=1.5) | `onedif_kernel` | Once-differentiable Matérn |
| Matérn (ν=2.5) | `Matern_2_5_kernel` | Twice-differentiable Matérn |
| Squared Exponential | `squared_exponential_kernel` | Infinitely smooth (RBF) |
| General Matérn | `matern_kernel_factory(nu)` | Factory for arbitrary ν |

Kernel parameters `theta` are structured as `[alpha, length_scale]` where:
- `alpha`: Variance/scaling parameter
- `length_scale`: Characteristic length scale

## Project Structure

```
Dis_Spatial/
├── src/
│   ├── estimation_torch.py    # Core GPPEstimation class
│   ├── prediction.py          # GPPPrediction class
│   ├── kernel.py              # Kernel functions
│   ├── generation.py          # Data generation utilities
│   ├── networks.py            # Network topology generators
│   ├── weights.py             # Optimal weight computation
│   └── utils.py               # Helper functions
├── expriements/               
│   └── decentralized/         # Experiment scripts
│       ├── varying_parameter/
│       ├── varying_sample_size/
│       ├── varying_network/
│       └── ...
├── real_data/                 # Real data experiments
└── test/                      # Some tests
```

## Experiments


### Simulation Studies

The `expriements/decentralized/` directory contains reproducible experiments:

| Category | Experiment | Folder |
|----------|------------|--------|
| **Convergence** | Varying covariance parameters | `varying_parameter/` |
| | Data partitioning schemes | `partition/` |
| | Network connectivity | `varying_network/` |
| | Imbalanced sample sizes | `Unequal_sample_size/` |
| **Inference Accuracy** | Impact of sample size | `varying_sample_size/` |
| | Impact of rank | `varying_rank/` |
| | Confidence intervals | `CI/` |
| | Model misspecification | `misspecified/` |
| **Other** | Prediction accuracy | `prediction/` |
| | Computation efficiency | `time_comparison/` |
| | Smoothness parameter (ν) estimation | `estimating_nu/` |
| | Scaled Hessian positiveness | `HessianP/` |

### Real Data Experiments

The `real_data/` directory contains total precipitable water (TPW) satellite data experiments:

**First scenario** — Data cannot be transferred (privacy/cost constraints); each satellite corresponds to a machine:
| Experiment | Folder |
|------------|--------|
| Algorithm convergence & data visualization | `First_scenario/estimation/` |
| Confidence intervals | `First_scenario/CI/` |

**Second scenario** — Data transfer is feasible for improved computational efficiency:
| Experiment | Folder |
|------------|--------|
| Time comparison (varying machine count) | `Second_scenario/time_com_varying_J/` |
| Time comparison (varying rank) | `Second_scenario/time_com_varying_m/` |
| RMSPE (varying rank) | `Second_scenario/RMSPE_varying_m/` |

Each subfolder contains Python scripts to run experiments and Jupyter notebooks to generate plots.


## License

MIT License
