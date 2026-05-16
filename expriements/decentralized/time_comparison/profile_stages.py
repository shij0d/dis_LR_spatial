# -*- coding: utf-8 -*-
"""Profile each stage of ce_optimize_stage2 to find the real bottleneck."""
import sys, time, math
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')

import torch
from sklearn.gaussian_process.kernels import Matern
from src.kernel import exponential_kernel
from src.generation import GPPSampleGenerator
from src.estimation_torch import GPPEstimation, _K_and_dK_dl
from joblib import Parallel, delayed

# ── data setup ──────────────────────────────────────────────────────────────
alpha, length_scale, nu = 1, 0.1, 0.5
N = 80_000; m = 50
mis_dis = 0.02
l = math.sqrt(2 * N) * mis_dis
kernel = alpha * Matern(length_scale=length_scale, nu=nu)
sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=(-l/2,l/2,-l/2,l/2),
                              kernel=kernel, coefficients=(-1,2,3,-2,1),
                              noise=2, seed=2024)
data, knots = sampler.generate_obs_gpp(m=m, method="random")
knots = torch.tensor(knots, dtype=torch.float64)

# initial state (pretend these are reasonable estimates)
mu     = torch.zeros((m, 1), dtype=torch.float64)
Sigma  = torch.eye(m, dtype=torch.float64) * 0.01
beta   = torch.tensor([-1., 2., 3., -2., 1.], dtype=torch.float64).reshape(-1, 1)
delta  = torch.tensor([[0.25]], dtype=torch.float64)
theta  = torch.tensor([1.0, 0.1], dtype=torch.float64)

def bench(label, fn, reps=3):
    """Warm up once then average reps runs."""
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = (time.perf_counter() - t0) / reps
    print(f"  {label:<45s} {elapsed*1000:8.1f} ms")
    return elapsed

print(f"\nN={N}, m={m}  (all timings are per-call averages over 3 reps)")
print("="*65)

# ── raw baseline: single matmul and single _batch_theta call ──────────────
torch.set_num_threads(1)
full_data = torch.tensor(data[:, :2], dtype=torch.float64)
invK_base = torch.linalg.inv(exponential_kernel(knots, knots, theta))

print("\n--- Raw operation timings (J=1 equivalent, no dispatch overhead) ---")
bench("cdist(80000x2, 50x2)",
      lambda: torch.cdist(full_data, knots))
K_nl_raw = torch.cdist(full_data, knots)  # pre-compute for next benches
K_nl_raw = torch.exp(-K_nl_raw / 0.1)
bench("matmul (80000x50)@(50x50)  [= B = K_nl @ invK]",
      lambda: K_nl_raw @ invK_base)
B_raw = K_nl_raw @ invK_base
bench("B.T @ B  (50x80000)@(80000x50)",
      lambda: B_raw.T @ B_raw)

def single_batch_theta_call():
    errorV = torch.zeros(N, 1, dtype=torch.float64)  # dummy
    M = torch.eye(m, dtype=torch.float64) * 0.01
    tv_base = theta.squeeze()
    D_nl = torch.cdist(full_data, knots)
    D_nn = torch.cdist(knots, knots)

    def _grad(tv):
        K_nl, dK_nl_dl = _K_and_dK_dl(exponential_kernel, D_nl, tv)
        K_nn, dK_nn_dl = _K_and_dK_dl(exponential_kernel, D_nn, tv)
        invK = torch.linalg.inv(K_nn)
        B = K_nl @ invK
        G_B = 2 * delta * (B @ M + errorV @ mu.T)
        dF_dK_nl = G_B @ invK
        dF_dK_nn = -(B.T @ G_B @ invK)
        g_a = (torch.sum(dF_dK_nl * K_nl) + torch.sum(dF_dK_nn * K_nn)) / tv[0]
        g_l = torch.sum(dF_dK_nl * dK_nl_dl) + torch.sum(dF_dK_nn * dK_nn_dl)
        return torch.stack([g_a, g_l])

    g = _grad(tv_base)
    eps0 = max(tv_base[0].abs().item(), 1e-6) * 1e-4
    eps1 = max(tv_base[1].abs().item(), 1e-6) * 1e-4
    g_p0 = _grad(tv_base + tv_base.new_tensor([eps0, 0.0]))
    g_p1 = _grad(tv_base + tv_base.new_tensor([0.0, eps1]))
    h0 = torch.stack([(g_p0[0] - g[0]) / eps0, (g_p1[0] - g[0]) / eps1])
    h1 = torch.stack([(g_p0[1] - g[1]) / eps0, (g_p1[1] - g[1]) / eps1])
    return g, torch.stack([h0, h1])

bench("Full _batch_theta (N=80000, analytical+FD, sequential)",
      single_batch_theta_call)
print()

for J in [1, 2, 4, 8, 16]:
    weights  = torch.ones((J, J), dtype=torch.float64) / J
    dis_data = sampler.data_split(data, J)
    gpp = GPPEstimation(dis_data, exponential_kernel, knots, weights)
    torch.set_num_threads(1)

    # pre-extract (same as in the actual code)
    locs  = [dis_data[j][:, :2]              for j in range(J)]
    z     = [dis_data[j][:, 2].unsqueeze(1)  for j in range(J)]
    X     = [dis_data[j][:, 3:]              for j in range(J)]

    pool_blas  = Parallel(n_jobs=J, backend='threading')
    pool_theta = Parallel(n_jobs=J, backend='threading')

    # ── Stage A ──────────────────────────────────────────────────────────
    K    = exponential_kernel(knots, knots, theta)
    invK = torch.linalg.inv(K)

    def _stage_A(j, beta_j, invK_j):
        local_locs = dis_data[j][:, :2]
        local_z    = dis_data[j][:, 2].reshape(-1, 1)
        local_X    = dis_data[j][:, 3:]
        B = exponential_kernel(local_locs, knots, beta_j) @ invK_j
        errorV = local_X @ beta_j - local_z
        return -B.T @ errorV, B.T @ B, B

    print(f"\nJ={J}  (n_obs per machine = {N//J})")
    bench(f"Stage A (threading, J={J})",
          lambda: pool_blas(delayed(_stage_A)(j, beta, invK) for j in range(J)))

    # ── Theta worker (analytical + FD Hessian) ──────────────────────────
    def _batch_theta(j, locs_j, z_j, X_j, knots_, kfn, mu_j, Sigma_j, beta_j, delta_j, theta_j):
        torch.set_num_threads(1)
        n = z_j.shape[0]
        errorV = X_j @ beta_j - z_j
        M = Sigma_j + mu_j @ mu_j.T
        tv_base = theta_j.squeeze()
        D_nl = torch.cdist(locs_j, knots_)
        D_nn = torch.cdist(knots_, knots_)

        def _grad(tv):
            K_nl, dK_nl_dl = _K_and_dK_dl(kfn, D_nl, tv)
            K_nn, dK_nn_dl = _K_and_dK_dl(kfn, D_nn, tv)
            invK = torch.linalg.inv(K_nn)
            B = K_nl @ invK
            G_B = 2 * delta_j * (B @ M + errorV @ mu_j.T)
            dF_dK_nl = G_B @ invK
            dF_dK_nn = -(B.T @ G_B @ invK)
            g_a = (torch.sum(dF_dK_nl * K_nl) + torch.sum(dF_dK_nn * K_nn)) / tv[0]
            g_l = torch.sum(dF_dK_nl * dK_nl_dl) + torch.sum(dF_dK_nn * dK_nn_dl)
            return torch.stack([g_a, g_l])

        g = _grad(tv_base)
        eps0 = max(tv_base[0].abs().item(), 1e-6) * 1e-4
        eps1 = max(tv_base[1].abs().item(), 1e-6) * 1e-4
        g_p0 = _grad(tv_base + tv_base.new_tensor([eps0, 0.0]))
        g_p1 = _grad(tv_base + tv_base.new_tensor([0.0, eps1]))
        h0 = torch.stack([(g_p0[0] - g[0]) / eps0, (g_p1[0] - g[0]) / eps1])
        h1 = torch.stack([(g_p0[1] - g[1]) / eps0, (g_p1[1] - g[1]) / eps1])
        return g.reshape(-1, 1), torch.stack([h0, h1])

    bench(f"Theta stage (threading, J={J})",
          lambda: pool_theta(
              delayed(_batch_theta)(j, locs[j], z[j], X[j], knots, exponential_kernel,
                                   mu, Sigma, beta, delta, theta)
              for j in range(J)))

    # ── com_hessian_theta (sequential, on master) ─────────────────────────
    bench(f"com_hessian_theta (sequential, master)",
          lambda: gpp.com_hessian_theta(mu, Sigma, theta))

    # ── K + invK on master ───────────────────────────────────────────────
    bench(f"K + invK on master (sequential)",
          lambda: torch.linalg.inv(exponential_kernel(knots, knots, theta)))
