# -*- coding: utf-8 -*-
"""Benchmark Stage A: Python/joblib vs C++/OpenMP — all variants use pre-computed dist."""
import sys, time, math, os
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')

import torch
from sklearn.gaussian_process.kernels import Matern
from src.kernel import exponential_kernel
from src.generation import GPPSampleGenerator
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
from torch.utils.cpp_extension import load_inline

# ── C++ source ──────────────────────────────────────────────────────────────
cpp_source = """
#include <torch/torch.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ── C++ baseline with pre-computed dist ──────────────────────────────────
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, double>
stageA_omp_baseline_nodist_timed(
    const std::vector<torch::Tensor>& dist_vec,
    const std::vector<torch::Tensor>& z_vec,
    const std::vector<torch::Tensor>& X_vec,
    const torch::Tensor& invK,
    const torch::Tensor& beta,
    double alpha_val,
    double length_scale,
    int num_threads)
{
    int J = dist_vec.size();
    int m = invK.size(0);
    std::vector<torch::Tensor> y_mu_vec(J), y_Sigma_vec(J), B_vec(J);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        auto K_nl = alpha_val * torch::exp(-dist_vec[j] / length_scale);
        auto B = torch::matmul(K_nl, invK);
        auto errorV = torch::matmul(X_vec[j], beta) - z_vec[j];
        y_mu_vec[j] = -torch::matmul(B.t(), errorV);
        y_Sigma_vec[j] = torch::matmul(B.t(), B);
        B_vec[j] = B;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {y_mu_vec, y_Sigma_vec, B_vec, elapsed_ms};
}

// ── C++ optimized with pre-computed dist ──────────────────────────────────
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, double>
stageA_omp_opt_nodist_timed(
    const std::vector<torch::Tensor>& dist_vec,
    const std::vector<torch::Tensor>& z_vec,
    const std::vector<torch::Tensor>& X_vec,
    const torch::Tensor& invK,
    const torch::Tensor& beta,
    double alpha_val,
    double length_scale,
    int num_threads)
{
    int J = dist_vec.size();
    int m = invK.size(0);
    std::vector<torch::Tensor> y_mu_vec(J), y_Sigma_vec(J), B_vec(J);

    // Pre-allocate output buffers
    std::vector<torch::Tensor> K_nl_buf(J), B_buf(J), errorV_buf(J),
                               y_mu_buf(J), y_Sigma_buf(J);
    for (int j = 0; j < J; j++) {
        int n = dist_vec[j].size(0);
        auto opts = dist_vec[j].options();
        K_nl_buf[j]    = at::empty({n, m}, opts);
        B_buf[j]       = at::empty({n, m}, opts);
        errorV_buf[j]  = at::empty({n, 1}, opts);
        y_mu_buf[j]    = at::empty({m, 1}, opts);
        y_Sigma_buf[j] = at::empty({m, m}, opts);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        c10::InferenceMode guard;
        int n = dist_vec[j].size(0);

        // Fused exp: single memory pass into pre-allocated K_nl_buf
        double* out = K_nl_buf[j].mutable_data_ptr<double>();
        const double* inp = dist_vec[j].const_data_ptr<double>();
        int64_t N_elem = (int64_t)n * m;
        #pragma omp simd
        for (int64_t i = 0; i < N_elem; i++)
            out[i] = alpha_val * std::exp(-inp[i] / length_scale);

        torch::matmul_out(B_buf[j], K_nl_buf[j], invK);
        torch::addmm_out(errorV_buf[j], z_vec[j], X_vec[j], beta, -1.0, 1.0);
        torch::matmul_out(y_mu_buf[j], B_buf[j].t(), errorV_buf[j]);
        y_mu_buf[j].neg_();
        torch::matmul_out(y_Sigma_buf[j], B_buf[j].t(), B_buf[j]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int j = 0; j < J; j++) {
        y_mu_vec[j]    = y_mu_buf[j].clone();
        y_Sigma_vec[j] = y_Sigma_buf[j].clone();
        B_vec[j]       = B_buf[j].clone();
    }

    return {y_mu_vec, y_Sigma_vec, B_vec, elapsed_ms};
}

// ── C++ fused: pre-computes dist internally + pre-alloc + fused exp ───────
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, double>
stageA_omp_fused_timed(
    const std::vector<torch::Tensor>& locs_vec,
    const std::vector<torch::Tensor>& z_vec,
    const std::vector<torch::Tensor>& X_vec,
    const torch::Tensor& knots,
    const torch::Tensor& invK,
    const torch::Tensor& beta,
    double alpha_val,
    double length_scale,
    int num_threads)
{
    int J = locs_vec.size();
    int m = knots.size(0);

    // Pre-compute dist once (outside timer)
    std::vector<torch::Tensor> dist_buf(J);
    {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < J; j++) {
            c10::InferenceMode guard;
            dist_buf[j] = torch::cdist(locs_vec[j], knots);
        }
    }

    // Pre-allocate output buffers
    std::vector<torch::Tensor> K_nl_buf(J), B_buf(J), errorV_buf(J),
                               y_mu_buf(J), y_Sigma_buf(J);
    for (int j = 0; j < J; j++) {
        int n = locs_vec[j].size(0);
        auto opts = locs_vec[j].options();
        K_nl_buf[j]    = at::empty({n, m}, opts);
        B_buf[j]       = at::empty({n, m}, opts);
        errorV_buf[j]  = at::empty({n, 1}, opts);
        y_mu_buf[j]    = at::empty({m, 1}, opts);
        y_Sigma_buf[j] = at::empty({m, m}, opts);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        c10::InferenceMode guard;
        int n = locs_vec[j].size(0);

        // Fused exp: single memory pass
        double* out = K_nl_buf[j].mutable_data_ptr<double>();
        const double* inp = dist_buf[j].const_data_ptr<double>();
        int64_t N_elem = (int64_t)n * m;
        #pragma omp simd
        for (int64_t i = 0; i < N_elem; i++)
            out[i] = alpha_val * std::exp(-inp[i] / length_scale);

        torch::matmul_out(B_buf[j], K_nl_buf[j], invK);
        torch::addmm_out(errorV_buf[j], z_vec[j], X_vec[j], beta, -1.0, 1.0);
        torch::matmul_out(y_mu_buf[j], B_buf[j].t(), errorV_buf[j]);
        y_mu_buf[j].neg_();
        torch::matmul_out(y_Sigma_buf[j], B_buf[j].t(), B_buf[j]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<torch::Tensor> y_mu_vec(J), y_Sigma_vec(J), B_vec(J);
    for (int j = 0; j < J; j++) {
        y_mu_vec[j]    = y_mu_buf[j].clone();
        y_Sigma_vec[j] = y_Sigma_buf[j].clone();
        B_vec[j]       = B_buf[j].clone();
    }

    return {y_mu_vec, y_Sigma_vec, B_vec, elapsed_ms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stageA_omp_baseline_nodist_timed", &stageA_omp_baseline_nodist_timed,
          "C++ baseline with pre-computed dist");
    m.def("stageA_omp_opt_nodist_timed", &stageA_omp_opt_nodist_timed,
          "C++ optimized (InfMode+fused exp+prealloc) with pre-computed dist");
    m.def("stageA_omp_fused_timed", &stageA_omp_fused_timed,
          "C++ fused: pre-computes dist + fused exp + prealloc");
}
"""

# Compile C++ extension
print("Compiling C++ extension (one-time)...", flush=True)
try:
    ext = load_inline(
        name='stageA_cpp_ext',
        cpp_sources=cpp_source,
        extra_cflags=['-fopenmp', '-O3', '-march=native'],
        extra_ldflags=['-fopenmp'],
        with_cuda=False,
        verbose=False,
    )
except Exception:
    ext = load_inline(
        name='stageA_cpp_ext',
        cpp_sources=cpp_source,
        extra_cflags=['-fopenmp', '-O3'],
        extra_ldflags=['-fopenmp'],
        with_cuda=False,
        verbose=True,
    )
print("  compiled OK\n", flush=True)

# Enable CPU affinity for OpenMP: pin threads to physical cores, reduce barrier jitter
os.environ['OMP_PROC_BIND'] = 'true'
os.environ['OMP_PLACES'] = 'cores'

# ── Data setup ────────────────────────────────────────────────────────────────
alpha, length_scale, nu = 1, 0.1, 0.5
N = 80_000; m = 50; mis_dis = 0.02; l_ext = math.sqrt(2 * N) * mis_dis
kernel = alpha * Matern(length_scale=length_scale, nu=nu)
sampler = GPPSampleGenerator(num=N, min_dis=mis_dis, extent=(-l_ext/2,l_ext/2,-l_ext/2,l_ext/2),
                             kernel=kernel, coefficients=(-1,2,3,-2,1), noise=2, seed=2024)
data, knots = sampler.generate_obs_gpp(m=m, method="random")
knots_t = torch.tensor(knots, dtype=torch.float64)
beta  = torch.tensor([-1.,2.,3.,-2.,1.], dtype=torch.float64).reshape(-1,1)
theta = torch.tensor([1.0, 0.1], dtype=torch.float64)
invK  = torch.linalg.inv(exponential_kernel(knots_t, knots_t, theta))

# ── Python: pre-computed dist version ──────────────────────────────────────────
def stageA_py(dist_j, z_j, X_j):
    torch.set_num_threads(1)
    threadpool_limits(limits=1, user_api='blas').__enter__()
    K_nl = alpha * torch.exp(-dist_j / length_scale)
    B = K_nl @ invK
    errorV = X_j @ beta - z_j
    return -B.T @ errorV, B.T @ B, B

# ── Benchmark ──────────────────────────────────────────────────────────────────
print(f"{'J':>3} {'method':<32} {'wall(ms)':>9} {'ideal(ms)':>9} {'overhead':>9}  {'eff':>6}")
print("-" * 84)

for J in [1, 2, 4, 8, 16, 28]:
    dis_data = sampler.data_split(data, J)
    n = N // J
    locs_t = [torch.tensor(dis_data[j][:,:2], dtype=torch.float64) for j in range(J)]
    z_t    = [torch.tensor(dis_data[j][:,2], dtype=torch.float64).reshape(-1,1) for j in range(J)]
    X_t    = [torch.tensor(dis_data[j][:,3:], dtype=torch.float64) for j in range(J)]

    # Pre-compute dist for all chunks (once, same for all variants)
    torch.set_num_threads(1)
    threadpool_limits(limits=1, user_api='blas').__enter__()
    dist_t = [torch.cdist(locs_t[j], knots_t) for j in range(J)]

    # Sequential baseline (pre-computed dist)
    t0 = time.perf_counter()
    for _ in range(20):
        for j in range(J):
            stageA_py(dist_t[j], z_t[j], X_t[j])
    seq_total = (time.perf_counter() - t0) / 20
    ideal = seq_total / J

    # ── Python + joblib ──
    pool = Parallel(n_jobs=J, backend='threading')
    pool(delayed(stageA_py)(dist_t[j], z_t[j], X_t[j]) for j in range(J))  # warm
    t0 = time.perf_counter()
    for _ in range(20):
        pool(delayed(stageA_py)(dist_t[j], z_t[j], X_t[j]) for j in range(J))
    py_wall = (time.perf_counter() - t0) / 20

    # ── C++ baseline (ATen, pre-computed dist) ──
    _, _, _, cpp_base = ext.stageA_omp_baseline_nodist_timed(
        dist_t, z_t, X_t, invK, beta, alpha, length_scale, J)
    times = []
    for _ in range(20):
        _, _, _, t = ext.stageA_omp_baseline_nodist_timed(
            dist_t, z_t, X_t, invK, beta, alpha, length_scale, J)
        times.append(t)
    cpp_base_wall = sum(times) / len(times)

    # ── C++ optimized (pre-computed dist, pre-alloc, fused exp) ──
    _, _, _, cpp_opt = ext.stageA_omp_opt_nodist_timed(
        dist_t, z_t, X_t, invK, beta, alpha, length_scale, J)
    times = []
    for _ in range(20):
        _, _, _, t = ext.stageA_omp_opt_nodist_timed(
            dist_t, z_t, X_t, invK, beta, alpha, length_scale, J)
        times.append(t)
    cpp_opt_wall = sum(times) / len(times)

    # ── C++ fused (pre-computes dist internally + everything pre-alloc) ──
    _, _, _, cpp_fused = ext.stageA_omp_fused_timed(
        locs_t, z_t, X_t, knots_t, invK, beta, alpha, length_scale, J)
    times = []
    for _ in range(20):
        _, _, _, t = ext.stageA_omp_fused_timed(
            locs_t, z_t, X_t, knots_t, invK, beta, alpha, length_scale, J)
        times.append(t)
    cpp_fused_wall = sum(times) / len(times)

    def p(x): return f"{x:>8.1f}"
    def eff(w): return f"{ideal/w:.2f}x" if w > 0 else "inf"
    print(f"{J:>3} {'Python joblib':<32} {p(py_wall*1000)}  {p(ideal*1000)}  {p((py_wall-ideal)*1000)}ms  {eff(py_wall)}")
    print(f"{J:>3} {'C++ baseline (ATen)':<32} {p(cpp_base_wall)}  {p(ideal*1000)}  {p((cpp_base_wall/1000-ideal)*1000)}ms  {eff(cpp_base_wall/1000)}")
    print(f"{J:>3} {'C++ opt (InfMode+fused+prealloc)':<32} {p(cpp_opt_wall)}  {p(ideal*1000)}  {p((cpp_opt_wall/1000-ideal)*1000)}ms  {eff(cpp_opt_wall/1000)}")
    print(f"{J:>3} {'C++ fused (auto cdist+all opts)':<32} {p(cpp_fused_wall)}  {p(ideal*1000)}  {p((cpp_fused_wall/1000-ideal)*1000)}ms  {eff(cpp_fused_wall/1000)}")
    print()
