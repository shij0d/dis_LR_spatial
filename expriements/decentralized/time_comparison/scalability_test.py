# -*- coding: utf-8 -*-
"""
Scalability test: ce_optimize_stage2 wall time vs J (number of machines/threads).
Also monitors per-core CPU usage during each run.
"""
import os, sys, threading, time, math
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')

import torch
import psutil
from sklearn.gaussian_process.kernels import Matern
from src.estimation_torch import GPPEstimation
from src.generation import GPPSampleGenerator
from src.kernel import exponential_kernel

# ── setup ────────────────────────────────────────────────────────────────────
alpha, length_scale, nu = 1, 0.1, 0.5
N       = 80_000
T       = 5          # more iterations so loky worker startup is amortized
mis_dis = 0.02
l       = math.sqrt(2 * N) * mis_dis
extent  = -l/2, l/2, -l/2, l/2
coefficients = (-1, 2, 3, -2, 1)
noise_level  = 2

kernel  = alpha * Matern(length_scale=length_scale, nu=nu)
sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=extent,
                              kernel=kernel, coefficients=coefficients,
                              noise=noise_level, seed=2024)
data, knots = sampler.generate_obs_gpp(m=50, method="random")

# warm-up initial estimator on small data
print("Getting initial estimator ...", flush=True)
sampler0 = GPPSampleGenerator(num=2000, min_dis=mis_dis,
                               extent=(-l/2,l/2,-l/2,l/2),
                               kernel=kernel, coefficients=coefficients,
                               noise=noise_level, seed=3232134)
data0, knots0 = sampler0.generate_obs_gpp(m=50, method="random")
w0  = torch.ones((1, 1), dtype=torch.float64)
gpp0 = GPPEstimation(sampler0.data_split(data0, 1), exponential_kernel, knots0, w0)
beta0  = torch.tensor([-1, 2, 3, -2, 1], dtype=torch.float64)
delta0 = torch.tensor(0.25,  dtype=torch.float64)
theta0 = torch.tensor([alpha, length_scale], dtype=torch.float64)
x0 = gpp0.argument2vector_lik(beta0, delta0, theta0)
mu0, Sigma0, beta0, delta0, theta0, _ = gpp0.get_minimier(x0)
print("Initial estimator ready.\n", flush=True)

# ── CPU monitor thread ────────────────────────────────────────────────────────
class CPUMonitor:
    def __init__(self, interval=0.25):
        self.interval = interval
        self._stop   = threading.Event()
        self.samples = []
        self._t      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(percpu=True))
            time.sleep(self.interval)

    def report(self):
        if not self.samples:
            return "  no samples"
        import statistics
        ncores = len(self.samples[0])
        # average utilisation per core across all samples
        avg_per_core = [statistics.mean(s[c] for s in self.samples) for c in range(ncores)]
        busy = [u for u in avg_per_core if u > 5.0]
        lines = [
            f"  cores sampled : {ncores}",
            f"  cores active  : {len(busy)} / {ncores}  (avg util > 5%)",
            f"  mean util busy: {sum(busy)/len(busy):.1f}%  (of active cores)" if busy else "  all cores idle",
            f"  peak total util: {max(sum(s) for s in self.samples)/ncores:.1f}%",
        ]
        return "\n".join(lines)

monitor = CPUMonitor(interval=0.2)

# ── main sweep ────────────────────────────────────────────────────────────────
Js = [1, 2, 4, 8, 14, 20, 28]
Js = [1, 2]

print(f"{'J':>4}  {'wall(s)':>9}  {'speedup':>8}   CPU monitor")
print("-" * 70)

t_J1 = None
for J in Js:
    weights = torch.ones((J, J), dtype=torch.float64) / J
    dis_data = sampler.data_split(data, J)
    gpp = GPPEstimation(dis_data, exponential_kernel, knots, weights)

    monitor.start()
    t0 = time.perf_counter()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        gpp.ce_optimize_stage2(
            mu0, Sigma0, beta0, delta0, theta0,
            T, J, thread_num=1, backend='threading'
        )
    elapsed = time.perf_counter() - t0
    monitor.stop()

    if J == 1:
        t_J1 = elapsed
    speedup = t_J1 / elapsed if t_J1 else 1.0

    print(f"{J:>4}  {elapsed:>9.2f}s  {speedup:>7.2f}x")
    print(monitor.report())
    print()
