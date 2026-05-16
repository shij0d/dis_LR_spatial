# -*- coding: utf-8 -*-
"""C++ implementation of ce_optimize_stage2 — full algorithm in C++ with OpenMP.

Usage:
    from ce_stage2_cpp_impl import ce_optimize_stage2_cpp
    mu, Sigma, beta_list, delta_list, theta_list, s_list = ce_optimize_stage2_cpp(
        locs_list, z_list, X_list, knots, mu0, Sigma0, beta0, delta0, theta0,
        T=5, S=5, num_threads=8)
"""
import sys, os
sys.path.insert(0, '/home/shij0d/documents/dis_LR_spatial')
import torch
from torch.utils.cpp_extension import load_inline

# ── C++ source ──────────────────────────────────────────────────────────────
cpp_source = r"""
#include <torch/torch.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

// ── Kernel helpers ─────────────────────────────────────────────────────────

// Return (K, dK/dl) for exponential kernel: K = alpha * exp(-D / l)
inline std::pair<at::Tensor, at::Tensor> K_and_dK_dl_exp(
    const at::Tensor& D, double alpha, double l)
{
    auto K = alpha * at::exp(-D / l);
    auto dK_dl = K * D / (l * l);
    return {K, dK_dl};
}

// ── Worker functions ───────────────────────────────────────────────────────

// Stage A: returns (y_mu_j, y_Sigma_j, B_j) for chunk j
// dist is pre-computed cdist(locs, knots), reused across iterations
void stageA_worker(
    const at::Tensor& dist_j,      // pre-computed (n, m)
    const at::Tensor& z_j,         // (n, 1)
    const at::Tensor& X_j,         // (n, p) or empty
    const at::Tensor& invK,        // (m, m)
    const at::Tensor& beta_j,      // (p, 1)
    double alpha,
    double l,
    at::Tensor& y_mu_out,          // (m, 1) pre-allocated
    at::Tensor& y_Sigma_out,       // (m, m) pre-allocated
    at::Tensor& B_out)             // (n, m) pre-allocated
{
    c10::InferenceMode guard;
    int n = dist_j.size(0);
    int m = invK.size(0);

    // Fused exp into pre-allocated K_nl buffer
    auto K_nl = at::empty({n, m}, dist_j.options());
    {
        double* out = K_nl.mutable_data_ptr<double>();
        const double* inp = dist_j.const_data_ptr<double>();
        int64_t N = (int64_t)n * m;
        double inv_l = 1.0 / l;
        #pragma omp simd
        for (int64_t i = 0; i < N; i++)
            out[i] = alpha * std::exp(-inp[i] * inv_l);
    }

    // B = K_nl @ invK
    at::matmul_out(B_out, K_nl, invK);

    // errorV = X @ beta - z (or -z if no X)
    at::Tensor errorV;
    if (X_j.numel() > 0) {
        errorV = at::addmm(-z_j, X_j, beta_j, 1.0, -1.0);  // -z + X@beta = X@beta - z
    } else {
        errorV = -z_j;
    }

    // y_mu = -B.T @ errorV
    at::matmul_out(y_mu_out, B_out.t(), errorV);
    y_mu_out.neg_();

    // y_Sigma = B.T @ B
    at::matmul_out(y_Sigma_out, B_out.t(), B_out);
}

// Stage B: returns X_j.T @ (z_j - B_j @ mu) for chunk j
void stageB_worker(
    const at::Tensor& z_j,         // (n, 1)
    const at::Tensor& X_j,         // (n, p) or empty
    const at::Tensor& B_j,         // (n, m)
    const at::Tensor& mu,          // (m, 1)
    at::Tensor& y_beta_out)        // (p, 1) pre-allocated
{
    c10::InferenceMode guard;
    if (X_j.numel() == 0) return;
    // y_beta = X.T @ (z - B @ mu)
    auto residual = z_j - at::matmul(B_j, mu);
    at::matmul_out(y_beta_out, X_j.t(), residual);
}

// Stage C: returns scalar objective contribution for chunk j
at::Tensor stageC_worker(
    const at::Tensor& z_j,         // (n, 1)
    const at::Tensor& X_j,         // (n, p) or empty
    const at::Tensor& B_j,         // (n, m)
    const at::Tensor& mu,          // (m, 1)
    const at::Tensor& Sigma,       // (m, m)
    const at::Tensor& M,           // (m, m) = Sigma + mu @ mu.T — pre-computed
    const at::Tensor& beta_j)      // (p, 1)
{
    c10::InferenceMode guard;
    at::Tensor errorV;
    if (X_j.numel() > 0)
        errorV = at::matmul(X_j, beta_j) - z_j;
    else
        errorV = -z_j;

    // term1 = trace(B.T @ B @ M)
    auto BtB = at::matmul(B_j.t(), B_j);  // (m, m) — could reuse y_Sigma from stageA
    double term1 = at::trace(BtB.matmul(M)).item<double>();
    // term2 = 2 * errorV.T @ B @ mu
    double term2 = 2.0 * at::matmul(errorV.t(), at::matmul(B_j, mu)).item<double>();
    // term3 = errorV.T @ errorV
    double term3 = at::matmul(errorV.t(), errorV).item<double>();

    auto result = at::empty({1, 1}, z_j.options());
    result[0][0] = term1 + term2 + term3;
    return result;
}

// _batch_theta: gradient + Hessian for chunk j
// D_nl is pre-computed cdist(locs, knots)
// D_nn is pre-computed cdist(knots, knots) — same for all workers
void batch_theta_worker(
    const at::Tensor& D_nl_j,      // (n, m)
    const at::Tensor& D_nn,        // (m, m)
    const at::Tensor& z_j,         // (n, 1)
    const at::Tensor& X_j,         // (n, p) or empty
    const at::Tensor& mu,          // (m, 1)
    const at::Tensor& M,           // (m, m) = Sigma + mu @ mu.T
    const at::Tensor& beta_j,      // (p, 1)
    double delta_val,
    double alpha,
    double l,
    at::Tensor& grad_out,          // (2, 1) pre-allocated
    at::Tensor& hess_out)          // (2, 2) pre-allocated
{
    c10::InferenceMode guard;
    int n = D_nl_j.size(0);
    int m = D_nn.size(0);

    // errorV = X@beta - z
    at::Tensor errorV;
    if (X_j.numel() > 0)
        errorV = at::matmul(X_j, beta_j) - z_j;
    else
        errorV = -z_j;

    // Lambda: compute gradient for given theta = [a, l]
    auto compute_grad = [&](double a, double l_val) -> std::pair<double, double> {
        auto [K_nl, dK_nl_dl] = K_and_dK_dl_exp(D_nl_j, a, l_val);
        auto [K_nn, dK_nn_dl] = K_and_dK_dl_exp(D_nn, a, l_val);

        auto invK = at::linalg_inv(K_nn);
        auto B = at::matmul(K_nl, invK);

        // G_B = 2 * delta * (B @ M + errorV @ mu.T)
        auto G_B = 2.0 * delta_val * (at::matmul(B, M) + at::matmul(errorV, mu.t()));

        auto dF_dK_nl = at::matmul(G_B, invK);
        auto dF_dK_nn = -at::matmul(at::matmul(B.t(), G_B), invK);

        double g_a = (at::sum(dF_dK_nl * K_nl).item<double>() +
                      at::sum(dF_dK_nn * K_nn).item<double>()) / a;
        double g_l = at::sum(dF_dK_nl * dK_nl_dl).item<double>() +
                     at::sum(dF_dK_nn * dK_nn_dl).item<double>();
        return {g_a, g_l};
    };

    // Base gradient
    auto [g_a, g_l] = compute_grad(alpha, l);

    // FD Hessian
    double eps_a = std::max(std::abs(alpha), 1e-6) * 1e-4;
    double eps_l = std::max(std::abs(l), 1e-6) * 1e-4;

    auto [gp_a_a, gp_a_l] = compute_grad(alpha + eps_a, l);
    auto [gp_l_a, gp_l_l] = compute_grad(alpha, l + eps_l);

    // Fill outputs
    auto g_ptr = grad_out.mutable_data_ptr<double>();
    g_ptr[0] = g_a;  g_ptr[1] = g_l;

    auto h_ptr = hess_out.mutable_data_ptr<double>();
    // Row 0: dg_a/da, dg_a/dl
    h_ptr[0] = (gp_a_a - g_a) / eps_a;
    h_ptr[1] = (gp_l_a - g_a) / eps_l;
    // Row 1: dg_l/da, dg_l/dl
    h_ptr[2] = (gp_a_l - g_l) / eps_a;
    h_ptr[3] = (gp_l_l - g_l) / eps_l;
}

// ── Master functions ───────────────────────────────────────────────────────

// com_grad_theta: common gradient from prior term
at::Tensor com_grad_theta(
    const at::Tensor& knots,
    const at::Tensor& mu,
    const at::Tensor& Sigma,
    double alpha, double l)
{
    c10::InferenceMode guard;
    int m = knots.size(0);
    auto D_nn = at::cdist(knots, knots);
    auto [K, dK_dl] = K_and_dK_dl_exp(D_nn, alpha, l);
    auto [K_a, dK_a_dl] = K_and_dK_dl_exp(D_nn, alpha + 1e-6, l);
    // Use autograd-like approach: numerical gradient of common term
    // f = mu.T @ invK @ mu + trace(invK @ Sigma) + logdet(K)
    auto compute_f = [&](double a, double l_val) -> double {
        auto [Kv, _] = K_and_dK_dl_exp(D_nn, a, l_val);
        auto invKv = at::linalg_inv(Kv);
        double f1 = at::matmul(mu.t(), at::matmul(invKv, mu)).item<double>();
        double f2 = at::trace(at::matmul(invKv, Sigma)).item<double>();
        double f3 = at::logdet(Kv).item<double>();
        return f1 + f2 + f3;
    };
    double f0 = compute_f(alpha, l);
    double eps = 1e-6;
    double g_a = (compute_f(alpha + eps, l) - f0) / eps;
    double g_l = (compute_f(alpha, l + eps * 0.1) - f0) / (eps * 0.1);

    auto grad = at::empty({2, 1}, knots.options());
    grad[0][0] = g_a;
    grad[1][0] = g_l;
    return grad;
}

// com_hessian_theta: common Hessian from prior term (numerical)
at::Tensor com_hessian_theta(
    const at::Tensor& knots,
    const at::Tensor& mu,
    const at::Tensor& Sigma,
    double alpha, double l)
{
    c10::InferenceMode guard;
    int m = knots.size(0);
    auto D_nn = at::cdist(knots, knots);

    auto compute_f = [&](double a, double l_val) -> double {
        auto [Kv, _] = K_and_dK_dl_exp(D_nn, a, l_val);
        auto invKv = at::linalg_inv(Kv);
        double f1 = at::matmul(mu.t(), at::matmul(invKv, mu)).item<double>();
        double f2 = at::trace(at::matmul(invKv, Sigma)).item<double>();
        double f3 = at::logdet(Kv).item<double>();
        return f1 + f2 + f3;
    };

    double f0 = compute_f(alpha, l);
    double eps_a = std::max(std::abs(alpha), 1e-6) * 1e-4;
    double eps_l = std::max(std::abs(l), 1e-6) * 1e-4;

    // Gradients at perturbed points for FD Hessian
    double f_a_p = compute_f(alpha + eps_a, l);
    double f_a_m = compute_f(alpha - eps_a, l);
    double f_l_p = compute_f(alpha, l + eps_l);
    double f_l_m = compute_f(alpha, l - eps_l);
    double f_al_pp = compute_f(alpha + eps_a, l + eps_l);

    double h_aa = (f_a_p - 2.0*f0 + f_a_m) / (eps_a * eps_a);
    double h_ll = (f_l_p - 2.0*f0 + f_l_m) / (eps_l * eps_l);
    double h_al = (f_al_pp - f_a_p - f_l_p + f0) / (eps_a * eps_l);

    auto hess = at::empty({2, 2}, knots.options());
    auto h = hess.mutable_data_ptr<double>();
    h[0] = h_aa;  h[1] = h_al;
    h[2] = h_al;  h[3] = h_ll;
    return hess;
}

} // anonymous namespace

// ── Main entry point ───────────────────────────────────────────────────────

std::tuple<at::Tensor, at::Tensor,
           std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<int>>
ce_optimize_stage2_cpp_impl(
    const std::vector<at::Tensor>& locs_vec,
    const std::vector<at::Tensor>& z_vec,
    const std::vector<at::Tensor>& X_vec,
    const at::Tensor& knots,
    const at::Tensor& mu_init,
    const at::Tensor& Sigma_init,
    const at::Tensor& beta_init,
    const at::Tensor& delta_init,
    const at::Tensor& theta_init,
    int T,
    int S_max,
    int num_threads)
{
    int J = locs_vec.size();
    int m = knots.size(0);
    int p = (X_vec.size() > 0 && X_vec[0].numel() > 0) ? X_vec[0].size(1) : 0;
    double alpha = theta_init[0].item<double>();
    double l = theta_init[1].item<double>();
    double delta_val = delta_init[0][0].item<double>();

    auto opts = knots.options();
    auto mu = mu_init.clone();
    auto Sigma = Sigma_init.clone();

    // ── Pre-compute once ───────────────────────────────────────────────────
    // D_nn: pairwise distances between knots (same for all workers, all iterations)
    auto D_nn = at::cdist(knots, knots);  // (m, m)

    // dist_j: cdist(locs_j, knots) for each chunk (same for all iterations)
    std::vector<at::Tensor> dist_vec(J);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        c10::InferenceMode guard;
        dist_vec[j] = at::cdist(locs_vec[j], knots);
    }

    // y_XTX: average X.T @ X across chunks
    at::Tensor y_XTX;
    if (p > 0) {
        y_XTX = at::zeros({p, p}, opts);
        for (int j = 0; j < J; j++) {
            y_XTX += at::matmul(X_vec[j].t(), X_vec[j]);
        }
        y_XTX /= J;
    }

    // Average local sample size
    double avg_n = 0;
    for (int j = 0; j < J; j++) avg_n += locs_vec[j].size(0);
    avg_n /= J;

    // ── Pre-allocate per-worker buffers ────────────────────────────────────
    // Stage A buffers
    std::vector<at::Tensor> A_y_mu(J), A_y_Sigma(J), A_B(J);
    for (int j = 0; j < J; j++) {
        int n = locs_vec[j].size(0);
        A_y_mu[j]    = at::empty({m, 1}, opts);
        A_y_Sigma[j] = at::empty({m, m}, opts);
        A_B[j]       = at::empty({n, m}, opts);
    }

    // Stage B buffers
    std::vector<at::Tensor> B_y_beta(J);
    for (int j = 0; j < J; j++)
        B_y_beta[j] = at::empty({p, 1}, opts);

    // Theta buffers
    std::vector<at::Tensor> T_grad(J), T_hess(J);
    for (int j = 0; j < J; j++) {
        T_grad[j] = at::empty({2, 1}, opts);
        T_hess[j] = at::empty({2, 2}, opts);
    }

    // ── Output accumulators ────────────────────────────────────────────────
    std::vector<at::Tensor> beta_list, delta_list, theta_list;
    std::vector<int> s_list;
    beta_list.push_back(beta_init.clone());
    delta_list.push_back(delta_init.clone());
    theta_list.push_back(theta_init.clone());

    at::Tensor beta = beta_init.clone();
    at::Tensor theta = theta_init.clone();

    // Reusable workspace
    auto y_mu_agg    = at::empty({m, 1}, opts);
    auto y_Sigma_agg = at::empty({m, m}, opts);
    auto y_beta_agg  = at::empty({p, 1}, opts);
    auto y_theta_agg = at::empty({2, 1}, opts);
    auto y_hess_agg  = at::empty({2, 2}, opts);

    // ── Main loop ──────────────────────────────────────────────────────────
    for (int t = 0; t < T; t++) {
        alpha = theta[0].item<double>();
        l     = theta[1].item<double>();

        // K, invK on master
        auto [K_nn, _] = K_and_dK_dl_exp(D_nn, alpha, l);
        auto invK = at::linalg_inv(K_nn);

        // ── Stage A: parallel ──────────────────────────────────────────────
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < J; j++) {
            stageA_worker(dist_vec[j], z_vec[j], X_vec[j], invK, beta,
                         alpha, l, A_y_mu[j], A_y_Sigma[j], A_B[j]);
        }

        // Aggregate Stage A
        y_mu_agg.zero_(); y_Sigma_agg.zero_();
        for (int j = 0; j < J; j++) {
            y_mu_agg += A_y_mu[j];
            y_Sigma_agg += A_y_Sigma[j];
        }
        y_mu_agg /= J;  y_Sigma_agg /= J;

        // mu, Sigma update
        auto inv_term_A = at::linalg_inv(delta_val * J * y_Sigma_agg + invK);
        Sigma = inv_term_A;
        auto inv_term_mu = at::linalg_inv(delta_val * y_Sigma_agg + invK / J);
        mu = at::matmul(inv_term_mu, delta_val * y_mu_agg);

        // ── Stage B: parallel ──────────────────────────────────────────────
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < J; j++) {
            stageB_worker(z_vec[j], X_vec[j], A_B[j], mu, B_y_beta[j]);
        }

        // Aggregate Stage B
        y_beta_agg.zero_();
        for (int j = 0; j < J; j++) y_beta_agg += B_y_beta[j];
        y_beta_agg /= J;

        if (p > 0)
            beta = at::linalg_solve(y_XTX, y_beta_agg);
        beta_list.push_back(beta.clone());

        // ── Stage C: parallel (returns scalar per chunk) ───────────────────
        auto M = Sigma + at::matmul(mu, mu.t());
        double y_delta_sum = 0;
        #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:y_delta_sum)
        for (int j = 0; j < J; j++) {
            auto contrib = stageC_worker(z_vec[j], X_vec[j], A_B[j], mu, Sigma, M, beta);
            y_delta_sum += contrib[0][0].item<double>();
        }
        y_delta_sum /= J;
        delta_val = avg_n / y_delta_sum;

        auto delta_tensor = at::empty({1, 1}, opts);
        delta_tensor[0][0] = delta_val;
        delta_list.push_back(delta_tensor);

        // ── Theta inner loop ────────────────────────────────────────────────
        int s_used = 0;
        for (int s = 0; s < S_max; s++) {
            alpha = theta[0].item<double>();
            l     = theta[1].item<double>();

            // Parallel gradient + Hessian across chunks
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int j = 0; j < J; j++) {
                batch_theta_worker(dist_vec[j], D_nn, z_vec[j], X_vec[j],
                                  mu, M, beta, delta_val, alpha, l,
                                  T_grad[j], T_hess[j]);
            }

            // Aggregate
            y_theta_agg.zero_(); y_hess_agg.zero_();
            for (int j = 0; j < J; j++) {
                y_theta_agg += T_grad[j];
                y_hess_agg += T_hess[j];
            }
            y_theta_agg /= J;  y_hess_agg /= J;

            // Common gradient + Hessian (master, sequential)
            auto com_g = com_grad_theta(knots, mu, Sigma, alpha, l);
            auto com_h = com_hessian_theta(knots, mu, Sigma, alpha, l);

            auto grad = y_theta_agg * J + com_g;
            if (at::norm(grad).item<double>() < 1e-4) break;

            auto hess = y_hess_agg * J + com_h;
            auto [eigvals, eigvecs] = at::linalg_eigh(hess);
            auto abs_ev = at::abs(eigvals);
            double threshold = 0.01;
            auto mod_ev = at::where(abs_ev < threshold,
                                    at::scalar_tensor(threshold, opts),
                                    abs_ev);
            auto mod_hess = at::matmul(eigvecs, at::matmul(at::diag(mod_ev), eigvecs.t()));

            auto invh_grad = at::linalg_solve(mod_hess, grad);
            if (at::norm(invh_grad).item<double>() < 1e-5) break;

            double step = 0.4;
            theta = theta - step * invh_grad;
            s_used = s + 1;
        }

        theta_list.push_back(theta.clone());
        s_list.push_back(s_used);
    }

    return {mu, Sigma, beta_list, delta_list, theta_list, s_list};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ce_optimize_stage2_cpp_impl", &ce_optimize_stage2_cpp_impl,
          "Full ce_optimize_stage2 in C++ with OpenMP");
}
"""

# ── Compile ─────────────────────────────────────────────────────────────────
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        os.environ.setdefault('OMP_PROC_BIND', 'true')
        os.environ.setdefault('OMP_PLACES', 'cores')
        try:
            _ext = load_inline(
                name='ce_stage2_cpp_ext',
                cpp_sources=cpp_source,
                extra_cflags=['-fopenmp', '-O3', '-march=native'],
                extra_ldflags=['-fopenmp'],
                with_cuda=False, verbose=False)
        except Exception:
            _ext = load_inline(
                name='ce_stage2_cpp_ext',
                cpp_sources=cpp_source,
                extra_cflags=['-fopenmp', '-O3'],
                extra_ldflags=['-fopenmp'],
                with_cuda=False, verbose=True)
    return _ext


def ce_optimize_stage2_cpp(locs_list, z_list, X_list, knots,
                           mu0, Sigma0, beta0, delta0, theta0,
                           T=5, S=5, num_threads=8):
    """C++ implementation of ce_optimize_stage2.

    Args:
        locs_list: list of J tensors, each (n_j, 2) — observation locations
        z_list:    list of J tensors, each (n_j, 1) — observation values
        X_list:    list of J tensors, each (n_j, p) — covariates (or empty tensors)
        knots:     (m, 2) tensor — knot locations
        mu0:       (m, 1) initial mu
        Sigma0:    (m, m) initial Sigma
        beta0:     (p, 1) initial beta
        delta0:    (1, 1) initial delta
        theta0:    (2,) or (2,1) initial theta [alpha, length_scale]
        T:         number of outer iterations
        S:         max Newton steps for theta inner loop
        num_threads: number of OpenMP threads

    Returns:
        mu, Sigma, beta_list, delta_list, theta_list, s_list
    """
    ext = _get_ext()

    # Ensure tensors are float64, reshape as needed
    def to_f64(t):
        return t.to(dtype=torch.float64).contiguous()

    locs_vec = [to_f64(t) for t in locs_list]
    z_vec    = [to_f64(t) for t in z_list]
    X_vec    = [to_f64(t) if t.numel() > 0 else torch.empty(0, dtype=torch.float64)
                for t in X_list]
    knots_t  = to_f64(knots)
    mu0_t    = to_f64(mu0).reshape(-1, 1)
    Sigma0_t = to_f64(Sigma0)
    beta0_t  = to_f64(beta0).reshape(-1, 1)
    delta0_t = to_f64(delta0).reshape(1, 1)
    theta0_t = to_f64(theta0).reshape(-1)

    mu, Sigma, beta_list, delta_list, theta_list, s_list = ext.ce_optimize_stage2_cpp_impl(
        locs_vec, z_vec, X_vec, knots_t, mu0_t, Sigma0_t, beta0_t, delta0_t, theta0_t,
        T, S, num_threads)

    return mu, Sigma, beta_list, delta_list, theta_list, s_list
