# -*- coding: utf-8 -*-
"""Scalability test for the C++ ce_optimize_stage2 backend.

Sweeps J = [1, 2, 4, 8, 16, 28] with num_threads = J on the same N=80_000 data,
measures wall time and reports speedup / parallel efficiency.

Set CE_DTYPE=fp32 to use float32. Default is float64.
"""
import os, sys, time, math, threading, statistics
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')

# Pin OMP threads to cores before importing torch (matters for libgomp/libiomp)
os.environ.setdefault('OMP_PROC_BIND', 'true')
os.environ.setdefault('OMP_PLACES', 'cores')

import torch
torch.set_num_threads(1)  # Pin BLAS to 1 per worker so OMP scaling is fair
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
except ImportError:
    pass
import psutil
from sklearn.gaussian_process.kernels import Matern
from src.generation import GPPSampleGenerator
from src.estimation_torch_cpp import ce_optimize_stage2_cpp

_DTYPE = torch.float32 if os.environ.get('CE_DTYPE', '').lower() in ('fp32', 'float32') else torch.float64

# ── Data setup ──────────────────────────────────────────────────────────────
alpha, length_scale, nu = 1, 0.1, 0.5
N       = 80_000
T       = 3
mis_dis = 0.02
l       = math.sqrt(2 * N) * mis_dis
extent  = -l/2, l/2, -l/2, l/2

kernel = alpha * Matern(length_scale=length_scale, nu=nu)
sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=extent,
                              kernel=kernel,
                              coefficients=(-1, 2, 3, -2, 1),
                              noise=2, seed=2024)
data, knots = sampler.generate_obs_gpp(m=50, method="random")

knots_t = torch.tensor(knots, dtype=torch.float64)
m_knots = knots_t.shape[0]
beta0   = torch.tensor([-1., 2., 3., -2., 1.], dtype=torch.float64).reshape(-1, 1)
delta0  = torch.tensor([[0.25]], dtype=torch.float64)
theta0  = torch.tensor([1.0, 0.1], dtype=torch.float64)
mu0     = torch.zeros((m_knots, 1), dtype=torch.float64)
Sigma0  = torch.eye(m_knots, dtype=torch.float64) * 0.01


# ── CPU monitor thread ──────────────────────────────────────────────────────
class CPUMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self._stop = threading.Event()
        self.samples = []
        self._t = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(percpu=True))
            time.sleep(self.interval)

    def report(self):
        if not self.samples:
            return "  no samples"
        ncores = len(self.samples[0])
        avg_per_core = [statistics.mean(s[c] for s in self.samples) for c in range(ncores)]
        busy = [u for u in avg_per_core if u > 5.0]
        return (f"  cores active: {len(busy)}/{ncores} (>5% util)"
                f"   peak total util: {max(sum(s) for s in self.samples)/ncores:.1f}%")


# ── Helper: build chunked tensors for a given J ─────────────────────────────
def split_for_J(J):
    dis = sampler.data_split(data, J)
    locs = [torch.tensor(dis[j][:, :2], dtype=torch.float64) for j in range(J)]
    zs   = [torch.tensor(dis[j][:, 2], dtype=torch.float64).reshape(-1, 1) for j in range(J)]
    Xs   = [torch.tensor(dis[j][:, 3:], dtype=torch.float64) for j in range(J)]
    return locs, zs, Xs


# Js to sweep
Js = [1, 2, 4, 8, 16, 28]

# Warm-up: compile the C++ extension and prime allocators
print("Warming up (compile + first run)...", flush=True)
locs1, zs1, Xs1 = split_for_J(1)
ce_optimize_stage2_cpp(locs1, zs1, Xs1, knots_t,
                       mu0, Sigma0, beta0, delta0, theta0,
                       T=1, S=2, num_threads=1, dtype=_DTYPE)
print("Ready.\n", flush=True)

monitor = CPUMonitor(interval=0.2)

print(f"N={N}  T={T}  dtype={_DTYPE}  (S_max=5, hessian=analytical)")
print(f"{'J':>3}  {'wall(s)':>9}  {'speedup':>8}  {'eff':>6}   CPU")
print("-" * 80)

t_J1 = None
for J in Js:
    locs, zs, Xs = split_for_J(J)

    # Warm-up at this J (compile-cached but allocator may need to warm)
    ce_optimize_stage2_cpp(locs, zs, Xs, knots_t,
                           mu0, Sigma0, beta0, delta0, theta0,
                           T=1, S=2, num_threads=J, dtype=_DTYPE)

    monitor.start()
    t0 = time.perf_counter()
    ce_optimize_stage2_cpp(locs, zs, Xs, knots_t,
                           mu0, Sigma0, beta0, delta0, theta0,
                           T=T, S=5, num_threads=J, dtype=_DTYPE)
    elapsed = time.perf_counter() - t0
    monitor.stop()

    if J == 1:
        t_J1 = elapsed
    speedup = t_J1 / elapsed
    eff = speedup / J

    print(f"{J:>3}  {elapsed:>8.2f}s  {speedup:>7.2f}x  {eff*100:>5.1f}% {monitor.report()}")
