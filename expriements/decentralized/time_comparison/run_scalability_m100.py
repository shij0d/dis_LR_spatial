"""Scalability sweep for m=100, FP64 and FP32."""
import os, sys, time, math, threading, statistics
os.environ.setdefault('OMP_PROC_BIND', 'true')
os.environ.setdefault('OMP_PLACES', 'cores')
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')

import torch
torch.set_num_threads(1)
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
except ImportError: pass
from sklearn.gaussian_process.kernels import Matern
from src.generation import GPPSampleGenerator
from src.estimation_torch_cpp import ce_optimize_stage2_cpp

_DTYPE = torch.float32 if os.environ.get('CE_DTYPE','').lower() in ('fp32','float32') else torch.float64
M_KNOTS = 100
N = 80_000
T = 3
mis_dis = 0.02
l = math.sqrt(2 * N) * mis_dis

kernel = 1.0 * Matern(length_scale=0.1, nu=0.5)
sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=(-l/2,l/2,-l/2,l/2),
                              kernel=kernel, coefficients=(-1,2,3,-2,1),
                              noise=2, seed=2024)
data, knots = sampler.generate_obs_gpp(m=M_KNOTS, method="random")
knots_t = torch.tensor(knots, dtype=torch.float64)
mk = knots_t.shape[0]
beta0  = torch.tensor([-1.,2.,3.,-2.,1.], dtype=torch.float64).reshape(-1,1)
delta0 = torch.tensor([[0.25]], dtype=torch.float64)
theta0 = torch.tensor([1.0, 0.1], dtype=torch.float64)
mu0    = torch.zeros((mk, 1), dtype=torch.float64)
Sigma0 = torch.eye(mk, dtype=torch.float64) * 0.01

Js = [1, 2, 4, 8, 16, 28]
walls = {}

# warm-up
d1 = sampler.data_split(data, 1)
locs1 = [torch.tensor(d1[0][:,:2], dtype=torch.float64) for _ in range(1)]
zs1   = [torch.tensor(d1[0][:,2], dtype=torch.float64).reshape(-1,1) for _ in range(1)]
Xs1   = [torch.tensor(d1[0][:,3:], dtype=torch.float64) for _ in range(1)]
ce_optimize_stage2_cpp(locs1, zs1, Xs1, knots_t, mu0, Sigma0, beta0, delta0, theta0,
                       T=1, S=2, num_threads=1, dtype=_DTYPE)

dtname = 'float32' if _DTYPE == torch.float32 else 'float64'
print(f"m={M_KNOTS}  N={N}  T={T}  dtype={dtname}\n")

print(f"{'J':>3}  {'wall(s)':>9}  {'speedup':>8}  {'eff':>6}")
print("-" * 35)
for J in Js:
    d = sampler.data_split(data, J)
    locs = [torch.tensor(d[j][:,:2], dtype=torch.float64) for j in range(J)]
    zs   = [torch.tensor(d[j][:,2], dtype=torch.float64).reshape(-1,1) for j in range(J)]
    Xs   = [torch.tensor(d[j][:,3:], dtype=torch.float64) for j in range(J)]

    ce_optimize_stage2_cpp(locs, zs, Xs, knots_t, mu0, Sigma0, beta0, delta0, theta0,
                           T=1, S=2, num_threads=J, dtype=_DTYPE)

    t0 = time.perf_counter()
    ce_optimize_stage2_cpp(locs, zs, Xs, knots_t, mu0, Sigma0, beta0, delta0, theta0,
                           T=T, S=5, num_threads=J, dtype=_DTYPE)
    elapsed = time.perf_counter() - t0
    walls[J] = elapsed

    if J == 1:
        t1 = elapsed
    speedup = t1 / elapsed
    eff = speedup / J * 100
    print(f"{J:>3}  {elapsed:>8.2f}s  {speedup:>7.2f}x  {eff:>5.1f}%")
