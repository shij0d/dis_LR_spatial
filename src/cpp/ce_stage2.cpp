// -*- c++ -*-
// C++ backend for GPP estimation — full ce_optimize_stage2 loop.
// Compiled via torch.utils.cpp_extension.load_inline by estimation_torch_cpp.py.
// OpenMP parallelizes the outer loop across J data chunks. BLAS per-call thread
// count is forced to 1 inside the C++ entry point so OMP scaling is not
// disturbed by MKL nested threading.

#include <torch/torch.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

// ── SIMD helpers: replace at::sum(A * B) with a single-pass dot product ────
// Allocating A*B as a temporary (n,m) tensor wastes ~4 MB per call and thrashes
// PyTorch's caching allocator under heavy OMP contention. These helpers take
// raw pointers and avoid the temporary entirely.
//
// Templated on the storage type T (float32 or float64). The reduction uses a
// double accumulator regardless of T so summation error stays bounded for the
// big (n,m) loops where N ≈ 5×10⁵.

template <typename T>
inline double dot_sum(const T* __restrict__ a,
                      const T* __restrict__ b, int64_t N)
{
    // SIMD reduction in T (full vector width per slot — 4× FP64 or 8× FP32 on
    // AVX2). Block size limits the per-block accumulation error, then we
    // promote each block sum to double. For BLOCK=4096 in FP32 the per-block
    // worst-case relative error is ε·√4096 ≈ 6e-5, the across-block sum is
    // exact in double → final relative error well below 1e-4.
    constexpr int64_t BLOCK = 4096;
    double s = 0.0;
    int64_t i = 0;
    for (; i + BLOCK <= N; i += BLOCK) {
        T block = (T)0;
        #pragma omp simd reduction(+:block)
        for (int64_t k = 0; k < BLOCK; k++)
            block += a[i + k] * b[i + k];
        s += (double)block;
    }
    if (i < N) {
        T tail = (T)0;
        const int64_t rem = N - i;
        #pragma omp simd reduction(+:tail)
        for (int64_t k = 0; k < rem; k++)
            tail += a[i + k] * b[i + k];
        s += (double)tail;
    }
    return s;
}

template <typename T>
inline double dot_sum_t(const at::Tensor& a, const at::Tensor& b) {
    return dot_sum<T>(a.const_data_ptr<T>(), b.const_data_ptr<T>(), a.numel());
}

// ── Exponential kernel helpers ─────────────────────────────────────────────

inline std::pair<at::Tensor, at::Tensor> K_and_dK_dl_exp(
    const at::Tensor& D, double alpha, double l)
{
    auto K = alpha * at::exp(-D / l);
    auto dK_dl = K * D / (l * l);
    return {K, dK_dl};
}

// ── Stage A worker (per chunk) ─────────────────────────────────────────────
// dist: pre-computed cdist(locs, knots), reused across iterations
// All outputs pre-allocated — zero allocations inside.

template <typename T>
void stageA_worker(
    const at::Tensor& dist,      // (n, m) pre-computed pairwise distances
    const at::Tensor& z,         // (n, 1)
    const at::Tensor& X,         // (n, p) or empty tensor
    const at::Tensor& invK,      // (m, m)
    const at::Tensor& beta,      // (p, 1)
    double alpha, double l,
    at::Tensor& y_mu_out,        // (m, 1) pre-allocated
    at::Tensor& y_Sigma_out,     // (m, m) pre-allocated
    at::Tensor& B_out)           // (n, m) pre-allocated
{
    c10::InferenceMode guard;
    int n = dist.size(0);
    int m = invK.size(0);
    T inv_l = (T)(1.0 / l);
    T a_T = (T)alpha;

    // Fused exp kernel: alpha * exp(-dist / l) in one memory pass
    auto K_nl = at::empty({n, m}, dist.options());
    {
        T* out = K_nl.mutable_data_ptr<T>();
        const T* inp = dist.const_data_ptr<T>();
        int64_t N = (int64_t)n * m;
        #pragma omp simd
        for (int64_t i = 0; i < N; i++)
            out[i] = a_T * std::exp(-inp[i] * inv_l);
    }

    // B = K_nl @ invK
    at::matmul_out(B_out, K_nl, invK);

    // errorV = X @ beta - z
    at::Tensor errorV;
    if (X.numel() > 0)
        errorV = at::addmm(-z, X, beta, 1.0, 1.0);  // = -z + X@beta = X@beta - z
    else
        errorV = -z;

    // y_mu = -B.T @ errorV
    at::matmul_out(y_mu_out, B_out.t(), errorV);
    y_mu_out.neg_();

    // y_Sigma = B.T @ B
    at::matmul_out(y_Sigma_out, B_out.t(), B_out);
}

// ── Stage B worker (per chunk) ─────────────────────────────────────────────
// Returns X.T @ (z - B @ mu) for OLS beta update.

void stageB_worker(
    const at::Tensor& z,         // (n, 1)
    const at::Tensor& X,         // (n, p) or empty
    const at::Tensor& B,         // (n, m)
    const at::Tensor& mu,        // (m, 1)
    at::Tensor& y_beta_out)      // (p, 1) pre-allocated
{
    c10::InferenceMode guard;
    if (X.numel() == 0) return;
    auto residual = z - at::matmul(B, mu);
    at::matmul_out(y_beta_out, X.t(), residual);
}

// ── Stage C worker (per chunk) ─────────────────────────────────────────────
// Returns scalar contribution to delta objective.
// y_Sigma = B.T @ B is passed from Stage A (avoid recomputation).

double stageC_worker(
    const at::Tensor& z,         // (n, 1)
    const at::Tensor& X,         // (n, p) or empty
    const at::Tensor& B,         // (n, m)
    const at::Tensor& mu,        // (m, 1)
    const at::Tensor& M,         // (m, m) = Sigma + mu @ mu.T
    const at::Tensor& beta,      // (p, 1)
    const at::Tensor& y_Sigma)   // (m, m) = B.T @ B (from Stage A)
{
    c10::InferenceMode guard;
    at::Tensor errorV;
    if (X.numel() > 0)
        errorV = at::matmul(X, beta) - z;
    else
        errorV = -z;

    double term1 = at::trace(at::matmul(y_Sigma, M)).item().toDouble();
    double term2 = 2.0 * at::matmul(errorV.t(), at::matmul(B, mu)).item().toDouble();
    double term3 = at::matmul(errorV.t(), errorV).item().toDouble();
    return term1 + term2 + term3;
}

// ── Theta gradient + FD Hessian worker (per chunk) ─────────────────────────
// Pre-allocated buffers (4× n×m) eliminate allocator contention.
// Buffer schedule: buf0=K_nl, buf1=dK_nl_dl, buf2=B/G_B workspace, buf3=dF_dK_nl
// All (m,m) tensors are tiny and allocated dynamically.

template <typename T>
void batch_theta_worker(
    const at::Tensor& D_nl,      // (n, m)
    const at::Tensor& D_nn,      // (m, m)
    const at::Tensor& z,         // (n, 1)
    const at::Tensor& X,         // (n, p) or empty
    const at::Tensor& mu,        // (m, 1)
    const at::Tensor& M,         // (m, m) = Sigma + mu @ mu.T
    const at::Tensor& beta,      // (p, 1)
    double delta_val,
    double alpha, double l,
    at::Tensor& grad_out,        // (2, 1) pre-allocated
    at::Tensor& hess_out,        // (2, 2) pre-allocated
    at::Tensor& buf0,            // (n, m) pre-alloc — holds K_nl (kept)
    at::Tensor& buf1,            // (n, m) pre-alloc — dK_nl_dl then dF_dK_nl
    at::Tensor& buf2,            // (n, m) pre-alloc — B then G_B
    at::Tensor& buf3,            // (n, m) pre-alloc — B@M workspace
    at::Tensor& errV_buf,        // (n, 1) pre-alloc — errorV
    at::Tensor& mm0,             // (m, m) pre-alloc — K_nn
    at::Tensor& mm1,             // (m, m) pre-alloc — dK_nn_dl
    at::Tensor& mm2,             // (m, m) pre-alloc — invK
    at::Tensor& mm3,             // (m, m) pre-alloc — BtG
    at::Tensor& mm4,             // (m, m) pre-alloc — dF_dK_nn
    at::Tensor& mm5)             // (m, m) pre-alloc — 2*delta*invK
{
    c10::InferenceMode guard;
    int n = D_nl.size(0);
    int m = D_nn.size(0);

    // errorV = X@beta - z  (or -z if no covariates) — pre-allocated
    if (X.numel() > 0) {
        at::addmm_out(errV_buf, z, X, beta, -1.0, 1.0);  // = -z + X@beta = X@beta - z
    } else {
        errV_buf.copy_(z);
        errV_buf.neg_();
    }

    int64_t N_nm = (int64_t)n * m;
    int64_t N_mm = (int64_t)m * m;

    // Analytical gradient. All workspace buffers come from caller (NUMA-local
    // to this worker thread, allocated inside an outer parallel region).
    auto compute_grad = [&](double a, double l_val) -> std::pair<double, double> {
        double inv_lv = 1.0 / l_val;
        double l2 = l_val * l_val;
        double inv_l2 = 1.0 / l2;
        T inv_lv_T = (T)inv_lv;
        T inv_l2_T = (T)inv_l2;
        T a_T = (T)a;

        // buf0 = K_nl = a * exp(-D_nl / l_val)
        {
            T* o = buf0.mutable_data_ptr<T>();
            const T* in_ = D_nl.const_data_ptr<T>();
            #pragma omp simd
            for (int64_t ii = 0; ii < N_nm; ii++)
                o[ii] = a_T * std::exp(-in_[ii] * inv_lv_T);
        }

        // mm0 = K_nn = a * exp(-D_nn / l_val)  (small, m=O(50))
        {
            T* o = mm0.mutable_data_ptr<T>();
            const T* in_ = D_nn.const_data_ptr<T>();
            #pragma omp simd
            for (int64_t ii = 0; ii < N_mm; ii++)
                o[ii] = a_T * std::exp(-in_[ii] * inv_lv_T);
        }
        // mm1 = dK_nn_dl = K_nn * D_nn / l²
        {
            T* o = mm1.mutable_data_ptr<T>();
            const T* k = mm0.const_data_ptr<T>();
            const T* d = D_nn.const_data_ptr<T>();
            #pragma omp simd
            for (int64_t ii = 0; ii < N_mm; ii++)
                o[ii] = k[ii] * d[ii] * inv_l2_T;
        }
        // mm2 = inv(K_nn)
        at::linalg_inv_out(mm2, mm0);

        // mm5 = 2*delta * invK  (precompute once so we avoid a SIMD scale pass
        // over the 4MB buf3 below — fold the scalar into the small (m,m) factor)
        {
            const T* in_ = mm2.const_data_ptr<T>();
            T* o = mm5.mutable_data_ptr<T>();
            T sc = (T)(2.0 * delta_val);
            #pragma omp simd
            for (int64_t ii = 0; ii < N_mm; ii++) o[ii] = sc * in_[ii];
        }

        // buf2 = B = K_nl @ invK
        at::matmul_out(buf2, buf0, mm2);

        // buf3 = (B @ M) + (errorV @ mu.T)   — UN-scaled; scaling folded into mm5
        at::matmul_out(buf3, buf2, M);
        at::addmm_out(buf3, buf3, errV_buf, mu.t(), 1.0, 1.0);

        // buf1 = G_B @ invK = buf3 @ (2*delta * invK) = buf3 @ mm5
        at::matmul_out(buf1, buf3, mm5);

        // mm3 = B.T @ buf3  (un-scaled), then mm4 = -(mm3 @ mm5)
        // → mm4 = -(B.T @ G_B @ invK)
        at::matmul_out(mm3, buf2.t(), buf3);
        at::matmul_out(mm4, mm3, mm5);
        {
            T* o = mm4.mutable_data_ptr<T>();
            #pragma omp simd
            for (int64_t ii = 0; ii < N_mm; ii++) o[ii] = -o[ii];
        }

        double s_nl_a = dot_sum_t<T>(buf1, buf0);
        // Build dK_nl_dl = K_nl * D_nl * inv_l2 into buf2 (buf2 is free here —
        // already consumed via matmul_out into mm3) and reduce against buf1.
        {
            T* o = buf2.mutable_data_ptr<T>();
            const T* knl = buf0.const_data_ptr<T>();
            const T* d = D_nl.const_data_ptr<T>();
            #pragma omp simd
            for (int64_t ii = 0; ii < N_nm; ii++)
                o[ii] = knl[ii] * d[ii] * inv_l2_T;
        }
        double s_nl_l = dot_sum_t<T>(buf1, buf2);
        double s_nn_a = dot_sum_t<T>(mm4, mm0);
        double s_nn_l = dot_sum_t<T>(mm4, mm1);
        double g_a = (s_nl_a + s_nn_a) / a;
        double g_l = s_nl_l + s_nn_l;
        return {g_a, g_l};
    };

    auto [g_a, g_l] = compute_grad(alpha, l);

    double eps_a = std::max(std::abs(alpha), 1e-6) * 1e-4;
    double eps_l = std::max(std::abs(l), 1e-6) * 1e-4;

    // ── α-perturbed gradient is analytic (no compute_grad call needed) ────
    // K(α, l) = α·E(l) is homogeneous in α of degree 1. Substituting α → α' in
    // every step of compute_grad shows:
    //   B(α', l)   = K_nl(α') @ invK_nn(α') = (α'/α)·K_nl @ (α/α')·invK = B
    //   G_B(α', l) = G_B    (depends only on B, M, errV, μ, δ)
    //   dF/dK_nl(α') = (1/α')·G_B @ invE_nn,  dF/dK_nn(α') likewise (1/α')-scaled
    //   ⇒ N(α', l) := Σ dF/dK_nl·K_nl + Σ dF/dK_nn·K_nn  is α-INVARIANT
    //   ⇒ g_a(α', l) = N(α, l)/α' = g_a(α, l)·(α/α')
    //   ⇒ g_l(α', l) = g_l(α, l)   (analogous derivation)
    // Forward FD then gives the closed forms below — equal to the exact
    // analytical second derivatives in the ε→0 limit and avoids one full
    // (n×m²) BLAS pipeline per Newton step (≈ 33% of theta time).
    double gp_a_a = g_a * (alpha / (alpha + eps_a));   // = g_a(α+ε_a, l)
    double gp_a_l = g_l;                                // = g_l(α+ε_a, l)

    auto [gp_l_a, gp_l_l] = compute_grad(alpha, l + eps_l);

    auto g_ptr = grad_out.mutable_data_ptr<T>();
    g_ptr[0] = (T)g_a;  g_ptr[1] = (T)g_l;

    auto h_ptr = hess_out.mutable_data_ptr<T>();
    h_ptr[0] = (T)((gp_a_a - g_a) / eps_a);
    h_ptr[1] = (T)((gp_l_a - g_a) / eps_l);
    h_ptr[2] = (T)((gp_a_l - g_l) / eps_a);
    h_ptr[3] = (T)((gp_l_l - g_l) / eps_l);
}

// ── Common gradient (prior term, O(m³), master only) ───────────────────────
// Analytical gradient of f = mu.T@invK@mu + trace(invK@Sigma) + logdet(K)
// w.r.t. exponential kernel K = alpha * exp(-D/l).
// dK/da = K/alpha,  dK/dl = K*D/(l*l)
// d(invK) = -invK @ dK @ invK
// d(logdet(K)) = trace(invK @ dK)

at::Tensor com_grad_theta(
    const at::Tensor& D_nn,      // (m, m)
    const at::Tensor& mu,        // (m, 1)
    const at::Tensor& Sigma,     // (m, m)
    double alpha, double l)
{
    c10::InferenceMode guard;
    // Always compute the (m,m) prior gradient in float64. m is small (≤ a few
    // hundred) so the upcast is negligible cost-wise and guards against the
    // float32 inverse of a poorly-conditioned K_nn.
    auto D_nn_d  = D_nn.scalar_type() == at::kDouble ? D_nn  : D_nn.to(at::kDouble);
    auto mu_d    = mu.scalar_type()    == at::kDouble ? mu    : mu.to(at::kDouble);
    auto Sigma_d = Sigma.scalar_type() == at::kDouble ? Sigma : Sigma.to(at::kDouble);

    auto [K, dK_dl] = K_and_dK_dl_exp(D_nn_d, alpha, l);
    auto dK_da = K / alpha;  // dK/d(alpha)
    auto invK = at::linalg_inv(K);

    // dF/dK = invK - invK @ (mu@mu.T + Sigma) @ invK
    auto M_common = at::matmul(mu_d, mu_d.t()) + Sigma_d;
    auto dF_dK = invK - at::matmul(at::matmul(invK, M_common), invK);

    double g_a = at::sum(dF_dK * dK_da).item().toDouble();
    double g_l = at::sum(dF_dK * dK_dl).item().toDouble();

    auto grad = at::empty({2, 1}, D_nn.options());  // input dtype
    if (grad.scalar_type() == at::kFloat) {
        auto gp = grad.mutable_data_ptr<float>();
        gp[0] = (float)g_a;  gp[1] = (float)g_l;
    } else {
        auto gp = grad.mutable_data_ptr<double>();
        gp[0] = g_a;  gp[1] = g_l;
    }
    return grad;
}

// ── Common Hessian (prior term, O(m³), master only) ────────────────────────
// Three modes:
//   "forward_fd": forward FD on analytical grad (fastest, O(eps) accurate)
//   "central_fd": central FD on analytical grad (2× cost, O(eps²) accurate)
//   "analytical": exact analytical Hessian (matches Python autograd)

at::Tensor com_hessian_theta(
    const at::Tensor& D_nn_arg,
    const at::Tensor& mu_arg,
    const at::Tensor& Sigma_arg,
    double alpha, double l,
    const std::string& mode)       // "forward_fd" | "central_fd" | "analytical"
{
    c10::InferenceMode guard;
    auto opts = D_nn_arg.options();   // input dtype (used only for output tensor)
    // Always run the (m,m) prior Hessian in double for stability.
    auto D_nn  = D_nn_arg.scalar_type()  == at::kDouble ? D_nn_arg  : D_nn_arg.to(at::kDouble);
    auto mu    = mu_arg.scalar_type()    == at::kDouble ? mu_arg    : mu_arg.to(at::kDouble);
    auto Sigma = Sigma_arg.scalar_type() == at::kDouble ? Sigma_arg : Sigma_arg.to(at::kDouble);
    auto S = at::matmul(mu, mu.t()) + Sigma;

    // ── Analytical mode: closed-form second derivatives ──────────────────
    // Verified against torch.autograd.functional.hessian to machine precision
    // (< 1e-13 relative error across 36 parameter combinations).
    if (mode == "analytical") {
        auto [K, dK_dl] = K_and_dK_dl_exp(D_nn, alpha, l);
        auto invK = at::linalg_inv(K);
        auto invK_S = at::matmul(invK, S);  // (m,m), reused below

        // h_aa = (2 * trace(invK @ S) - m) / alpha²
        double h_aa = (2.0 * at::trace(invK_S).item().toDouble() - D_nn.size(0))
                      / (alpha * alpha);

        // h_al = trace(invK @ S @ invK @ dK_dl) / alpha
        double h_al = at::trace(at::matmul(invK_S, at::matmul(invK, dK_dl)))
                      .item().toDouble() / alpha;

        // h_ll = trace(d(dF_dK)/dl @ dK_dl + dF_dK @ d²K_dl²)
        // dF_dK = invK - invK @ S @ invK
        // d(dF_dK)/dl = -invK @ dK_dl @ invK
        //               + invK @ dK_dl @ invK @ S @ invK
        //               + invK @ S @ invK @ dK_dl @ invK
        double l2 = l * l;
        auto d2K_dl2 = K * (D_nn * D_nn / (l2 * l2) - 2.0 * D_nn / (l * l2));
        auto dF_dK = invK - at::matmul(invK_S, invK);
        auto invK_dK = at::matmul(invK, dK_dl);  // invK @ dK_dl
        auto d_dF_dl = -at::matmul(invK_dK, invK)
                       + at::matmul(invK_dK, at::matmul(invK_S, invK))
                       + at::matmul(invK_S, at::matmul(invK, at::matmul(dK_dl, invK)));
        double h_ll = at::trace(at::matmul(d_dF_dl, dK_dl) +
                                at::matmul(dF_dK, d2K_dl2)).item().toDouble();

        auto hess = at::empty({2, 2}, opts);
        if (hess.scalar_type() == at::kFloat) {
            auto h = hess.mutable_data_ptr<float>();
            h[0] = (float)h_aa;  h[1] = (float)h_al;
            h[2] = (float)h_al;  h[3] = (float)h_ll;
        } else {
            auto h = hess.mutable_data_ptr<double>();
            h[0] = h_aa;  h[1] = h_al;
            h[2] = h_al;  h[3] = h_ll;
        }
        return hess;
    }

    // ── FD modes (forward or central) on the analytical gradient ─────────
    auto compute_g = [&](double a, double l_val) -> std::pair<double, double> {
        auto [Kv, dKv_dl] = K_and_dK_dl_exp(D_nn, a, l_val);
        auto dKv_da = Kv / a;
        auto invKv = at::linalg_inv(Kv);
        auto dF = invKv - at::matmul(at::matmul(invKv, S), invKv);
        double ga = at::sum(dF * dKv_da).item().toDouble();
        double gl = at::sum(dF * dKv_dl).item().toDouble();
        return {ga, gl};
    };

    auto [g0_a, g0_l] = compute_g(alpha, l);
    double eps_a = std::max(std::abs(alpha), 1e-6) * 1e-4;
    double eps_l = std::max(std::abs(l), 1e-6) * 1e-4;

    double h_aa, h_al, h_la, h_ll;

    if (mode == "central_fd") {
        auto [gp_a, gp_al_p]  = compute_g(alpha + eps_a, l);
        auto [gm_a, gm_al_m]  = compute_g(alpha - eps_a, l);
        auto [gp_la_p, gp_l]  = compute_g(alpha, l + eps_l);
        auto [gm_la_m, gm_l]  = compute_g(alpha, l - eps_l);
        h_aa = (gp_a - gm_a) / (2.0 * eps_a);
        h_al = (gp_al_p - gm_al_m) / (2.0 * eps_a);
        h_la = (gp_la_p - gm_la_m) / (2.0 * eps_l);
        h_ll = (gp_l - gm_l) / (2.0 * eps_l);
    } else {  // "forward_fd" (default)
        auto [gp_a, gp_al_p] = compute_g(alpha + eps_a, l);  // g_a(α+ε), g_l(α+ε)
        auto [gp_la_p, gp_l] = compute_g(alpha, l + eps_l);  // g_a(l+ε), g_l(l+ε)
        h_aa = (gp_a - g0_a) / eps_a;       // ∂g_a/∂α
        h_al = (gp_la_p - g0_a) / eps_l;    // ∂g_a/∂l
        h_la = (gp_al_p - g0_l) / eps_a;    // ∂g_l/∂α
        h_ll = (gp_l - g0_l) / eps_l;       // ∂g_l/∂l
    }

    auto hess = at::empty({2, 2}, opts);
    if (hess.scalar_type() == at::kFloat) {
        auto h = hess.mutable_data_ptr<float>();
        h[0] = (float)h_aa;  h[1] = (float)h_al;
        h[2] = (float)h_la;  h[3] = (float)h_ll;
    } else {
        auto h = hess.mutable_data_ptr<double>();
        h[0] = h_aa;  h[1] = h_al;
        h[2] = h_la;  h[3] = h_ll;
    }
    return hess;
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point: full ce_optimize_stage2 loop in C++
// ═══════════════════════════════════════════════════════════════════════════

std::tuple<at::Tensor, at::Tensor,
           std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>,
           std::vector<int>>
ce_optimize_stage2_cpp_impl(
    const std::vector<at::Tensor>& locs_vec,    // J tensors, each (n_j, 2)
    const std::vector<at::Tensor>& z_vec,       // J tensors, each (n_j, 1)
    const std::vector<at::Tensor>& X_vec,       // J tensors, each (n_j, p) or empty
    const at::Tensor& knots,                    // (m, 2)
    const at::Tensor& mu_init,                  // (m, 1)
    const at::Tensor& Sigma_init,               // (m, m)
    const at::Tensor& beta_init,                // (p, 1)
    const at::Tensor& delta_init,               // (1,) or (1,1)
    const at::Tensor& theta_init,               // (2,) or (2,1)
    int T, int S_max, int num_threads,
    const std::string& hessian_mode = "forward_fd")
{
    int J = locs_vec.size();
    int m = knots.size(0);
    int p = (!X_vec.empty() && X_vec[0].numel() > 0) ? X_vec[0].size(1) : 0;

    // ── Force BLAS to 1 thread per call ────────────────────────────────────
    // PyTorch is built with MKL which respects at::get_num_threads(). Without
    // this, MKL would spawn additional threads inside each OMP worker —
    // catastrophic oversubscription for J>1. Saved/restored at function exit.
    const int saved_threads = at::get_num_threads();
    at::set_num_threads(1);
    struct ThreadGuard {
        int prev;
        ~ThreadGuard() { at::set_num_threads(prev); }
    } _thread_guard{saved_threads};

    auto opts = knots.options();
    auto mu = mu_init.clone();
    auto Sigma = Sigma_init.clone();

    // ── Per-stage profiling (set CE_STAGE2_PROFILE=1 to enable) ────────────
    const char* prof_env = std::getenv("CE_STAGE2_PROFILE");
    const bool profile = (prof_env && prof_env[0] != '0');
    double t_stageA = 0, t_stageB = 0, t_stageC = 0, t_theta_par = 0;
    double t_aggA = 0, t_master_serial = 0, t_theta_serial = 0;
    auto tick = []() { return std::chrono::high_resolution_clock::now(); };
    auto tock = [](auto& acc, auto t0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // Scalar reads — handle both float32 and float64 inputs via item().toDouble().
    double alpha     = theta_init[0].item().toDouble();
    double l         = theta_init[1].item().toDouble();
    double delta_val = delta_init[0].item().toDouble();
    const bool is_float = (knots.scalar_type() == at::kFloat);

    // ── Pre-compute invariant quantities ───────────────────────────────────

    auto D_nn = at::cdist(knots, knots);  // (m, m), same for all workers/iterations

    // dist_vec[j] = cdist(locs_j, knots). The cdist output is allocated by the
    // worker thread itself → NUMA-local. Same OMP schedule(static) is used in
    // the hot loop so chunk j stays on the same worker across all iterations.
    std::vector<at::Tensor> dist_vec(J);
    // NUMA-local copy of input z and X — they come from Python allocated on
    // master's NUMA node and are read on every Stage A / B / C and theta inner
    // iter. Cloning inside the same OMP region (first-touch on the worker's
    // core) places them on the worker's socket and eliminates QPI traffic for
    // workers on socket 1.
    std::vector<at::Tensor> z_local(J), X_local(J);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        c10::InferenceMode guard;
        dist_vec[j] = at::cdist(locs_vec[j], knots);
        z_local[j]  = z_vec[j].contiguous().clone();
        X_local[j]  = (X_vec[j].numel() > 0) ? X_vec[j].contiguous().clone()
                                             : at::empty({0}, opts);
    }

    at::Tensor y_XTX;
    if (p > 0) {
        y_XTX = at::zeros({p, p}, opts);
        for (int j = 0; j < J; j++)
            y_XTX += at::matmul(X_local[j].t(), X_local[j]);
        y_XTX /= J;
    }

    double avg_n = 0;
    for (int j = 0; j < J; j++) avg_n += locs_vec[j].size(0);
    avg_n /= J;

    // ── Pre-allocate per-worker buffers, NUMA-local via first-touch ────────
    // Allocating inside an OMP parallel region with the same schedule used in
    // the hot loop ensures each worker's buffers land on its own socket's
    // memory bank (Linux first-touch policy). Touching with zero_() forces
    // the kernel to map pages immediately on the current core.

    std::vector<at::Tensor> A_y_mu(J), A_y_Sigma(J), A_B(J);
    std::vector<at::Tensor> B_y_beta(J);
    std::vector<at::Tensor> T_grad(J), T_hess(J);
    std::vector<at::Tensor> T_buf0(J), T_buf1(J), T_buf2(J), T_buf3(J);
    std::vector<at::Tensor> T_errV(J);
    std::vector<at::Tensor> T_mm0(J), T_mm1(J), T_mm2(J), T_mm3(J), T_mm4(J), T_mm5(J);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int j = 0; j < J; j++) {
        c10::InferenceMode guard;
        int n = locs_vec[j].size(0);
        A_y_mu[j]    = at::empty({m, 1}, opts);   A_y_mu[j].zero_();
        A_y_Sigma[j] = at::empty({m, m}, opts);   A_y_Sigma[j].zero_();
        A_B[j]       = at::empty({n, m}, opts);   A_B[j].zero_();
        B_y_beta[j]  = at::empty({p, 1}, opts);   if (p > 0) B_y_beta[j].zero_();
        T_grad[j]    = at::empty({2, 1}, opts);   T_grad[j].zero_();
        T_hess[j]    = at::empty({2, 2}, opts);   T_hess[j].zero_();
        T_buf0[j]    = at::empty({n, m}, opts);   T_buf0[j].zero_();
        T_buf1[j]    = at::empty({n, m}, opts);   T_buf1[j].zero_();
        T_buf2[j]    = at::empty({n, m}, opts);   T_buf2[j].zero_();
        T_buf3[j]    = at::empty({n, m}, opts);   T_buf3[j].zero_();
        T_errV[j]    = at::empty({n, 1}, opts);   T_errV[j].zero_();
        T_mm0[j]     = at::empty({m, m}, opts);   T_mm0[j].zero_();
        T_mm1[j]     = at::empty({m, m}, opts);   T_mm1[j].zero_();
        T_mm2[j]     = at::empty({m, m}, opts);   T_mm2[j].zero_();
        T_mm3[j]     = at::empty({m, m}, opts);   T_mm3[j].zero_();
        T_mm4[j]     = at::empty({m, m}, opts);   T_mm4[j].zero_();
        T_mm5[j]     = at::empty({m, m}, opts);   T_mm5[j].zero_();
    }

    // Aggregation buffers
    auto y_mu_agg    = at::empty({m, 1}, opts);
    auto y_Sigma_agg = at::empty({m, m}, opts);
    auto y_beta_agg  = at::empty({p, 1}, opts);
    auto y_theta_agg = at::empty({2, 1}, opts);
    auto y_hess_agg  = at::empty({2, 2}, opts);

    // ── Output lists ───────────────────────────────────────────────────────

    std::vector<at::Tensor> beta_list, delta_list, theta_list;
    std::vector<int> s_list;
    beta_list.push_back(beta_init.clone());
    delta_list.push_back(delta_init.clone());
    theta_list.push_back(theta_init.clone());

    at::Tensor beta = beta_init.clone();
    at::Tensor theta = theta_init.clone();

    // ═══════════════════════════════════════════════════════════════════════
    // Main optimization loop
    // ═══════════════════════════════════════════════════════════════════════

    for (int t = 0; t < T; t++) {
        alpha = theta[0].item().toDouble();
        l     = theta[1].item().toDouble();

        auto [K_nn, _] = K_and_dK_dl_exp(D_nn, alpha, l);
        auto invK = at::linalg_inv(K_nn);

        // ── Stage A: y_mu, y_Sigma, B (parallel across chunks) ────────
        auto _tA = tick();
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < J; j++) {
            if (is_float) {
                stageA_worker<float>(dist_vec[j], z_local[j], X_local[j], invK, beta,
                             alpha, l, A_y_mu[j], A_y_Sigma[j], A_B[j]);
            } else {
                stageA_worker<double>(dist_vec[j], z_local[j], X_local[j], invK, beta,
                             alpha, l, A_y_mu[j], A_y_Sigma[j], A_B[j]);
            }
        }
        if (profile) tock(t_stageA, _tA);

        auto _tAg = tick();
        y_mu_agg.zero_(); y_Sigma_agg.zero_();
        for (int j = 0; j < J; j++) {
            y_mu_agg += A_y_mu[j];
            y_Sigma_agg += A_y_Sigma[j];
        }
        y_mu_agg /= J;  y_Sigma_agg /= J;

        Sigma = at::linalg_inv(delta_val * J * y_Sigma_agg + invK);
        auto inv_mu = at::linalg_inv(delta_val * y_Sigma_agg + invK / J);
        mu = at::matmul(inv_mu, delta_val * y_mu_agg);
        if (profile) tock(t_aggA, _tAg);

        // ── Stage B: beta update (parallel across chunks) ──────────────
        auto _tB = tick();
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int j = 0; j < J; j++) {
            stageB_worker(z_local[j], X_local[j], A_B[j], mu, B_y_beta[j]);
        }

        y_beta_agg.zero_();
        for (int j = 0; j < J; j++) y_beta_agg += B_y_beta[j];
        y_beta_agg /= J;

        if (p > 0)
            beta = at::linalg_solve(y_XTX, y_beta_agg);
        beta_list.push_back(beta.clone());
        if (profile) tock(t_stageB, _tB);

        // ── Stage C: delta update (parallel across chunks) ─────────────
        auto _tC = tick();
        auto M = Sigma + at::matmul(mu, mu.t());

        double y_delta_sum = 0.0;
        #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:y_delta_sum)
        for (int j = 0; j < J; j++) {
            y_delta_sum += stageC_worker(z_local[j], X_local[j], A_B[j], mu, M, beta, A_y_Sigma[j]);
        }
        y_delta_sum /= J;
        delta_val = avg_n / y_delta_sum;
        if (profile) tock(t_stageC, _tC);

        auto delta_tensor = at::empty({1, 1}, opts);
        delta_tensor.fill_(delta_val);   // dtype-agnostic scalar write
        delta_list.push_back(delta_tensor);

        // ── Theta inner loop (modified Newton) ─────────────────────────
        int s_used = 0;
        for (int s = 0; s < S_max; s++) {
            alpha = theta[0].item().toDouble();
            l     = theta[1].item().toDouble();

            auto _tTp = tick();
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int j = 0; j < J; j++) {
                if (is_float) {
                    batch_theta_worker<float>(dist_vec[j], D_nn, z_local[j], X_local[j],
                                      mu, M, beta, delta_val, alpha, l,
                                      T_grad[j], T_hess[j],
                                      T_buf0[j], T_buf1[j], T_buf2[j], T_buf3[j],
                                      T_errV[j],
                                      T_mm0[j], T_mm1[j], T_mm2[j], T_mm3[j], T_mm4[j], T_mm5[j]);
                } else {
                    batch_theta_worker<double>(dist_vec[j], D_nn, z_local[j], X_local[j],
                                      mu, M, beta, delta_val, alpha, l,
                                      T_grad[j], T_hess[j],
                                      T_buf0[j], T_buf1[j], T_buf2[j], T_buf3[j],
                                      T_errV[j],
                                      T_mm0[j], T_mm1[j], T_mm2[j], T_mm3[j], T_mm4[j], T_mm5[j]);
                }
            }
            if (profile) tock(t_theta_par, _tTp);

            auto _tTs = tick();
            y_theta_agg.zero_(); y_hess_agg.zero_();
            for (int j = 0; j < J; j++) {
                y_theta_agg += T_grad[j];
                y_hess_agg += T_hess[j];
            }
            y_theta_agg /= J;  y_hess_agg /= J;

            auto com_g = com_grad_theta(D_nn, mu, Sigma, alpha, l);
            auto com_h = com_hessian_theta(D_nn, mu, Sigma, alpha, l, hessian_mode);
            if (profile) tock(t_theta_serial, _tTs);

            auto grad = y_theta_agg * J + com_g;
            if (at::linalg_norm(grad).item().toDouble() < 1e-4) break;

            auto hess = y_hess_agg * J + com_h;
            auto [eigvals, eigvecs] = at::linalg_eigh(hess);
            auto abs_ev = at::abs(eigvals);
            double threshold = 0.01;
            auto mod_ev = at::where(abs_ev < threshold,
                                    at::scalar_tensor(threshold, opts),
                                    abs_ev);
            auto mod_hess = at::matmul(eigvecs,
                                       at::matmul(at::diag(mod_ev), eigvecs.t()));

            auto invh_grad = at::linalg_solve(mod_hess, grad);
            if (at::linalg_norm(invh_grad).item().toDouble() < 1e-5) break;

            double step_size = 0.4;  // match Python ce_optimize_stage2
            theta = (theta - step_size * invh_grad.view({-1})).view({-1});
            s_used = s + 1;
        }

        theta_list.push_back(theta.clone());
        s_list.push_back(s_used);
    }

    if (profile) {
        double total = t_stageA + t_aggA + t_stageB + t_stageC +
                       t_theta_par + t_theta_serial;
        fprintf(stderr,
            "[ce_stage2 profile  J=%d num_threads=%d T=%d]\n"
            "  stage A parallel     : %8.2f ms\n"
            "  stage A aggregate+m/S: %8.2f ms\n"
            "  stage B (incl. solve): %8.2f ms\n"
            "  stage C (incl. red.) : %8.2f ms\n"
            "  theta parallel       : %8.2f ms  (%d outer x up to %d inner)\n"
            "  theta serial (com+agg): %7.2f ms\n"
            "  --------------------------------\n"
            "  measured total       : %8.2f ms\n",
            J, num_threads, T,
            t_stageA, t_aggA, t_stageB, t_stageC,
            t_theta_par, T, S_max, t_theta_serial,
            total);
        fflush(stderr);
    }
    return {mu, Sigma, beta_list, delta_list, theta_list, s_list};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ce_optimize_stage2_cpp_impl", &ce_optimize_stage2_cpp_impl,
          "Full ce_optimize_stage2 in C++ with OpenMP");
}
