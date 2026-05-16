# -*- coding: utf-8 -*-
"""Python wrapper for the C++ ce_optimize_stage2 backend.

Reads src/cpp/ce_stage2.cpp, compiles via torch.utils.cpp_extension.load_inline,
and exposes a single function ce_optimize_stage2_cpp() with the same signature
as the Python version in estimation_torch.py.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_CPP_PATH = os.path.join(os.path.dirname(__file__), 'cpp', 'ce_stage2.cpp')
_ext = None


def _get_ext():
    """Compile (once) and return the C++ extension module."""
    global _ext
    if _ext is not None:
        return _ext

    with open(_CPP_PATH, 'r') as f:
        cpp_source = f.read()

    # Prevent MKL from spawning its own thread pool.
    os.environ.setdefault('OMP_PROC_BIND', 'true')
    os.environ.setdefault('OMP_PLACES', 'cores')

    try:
        _ext = load_inline(
            name='ce_stage2_cpp_ext',
            cpp_sources=cpp_source,
            extra_cflags=['-fopenmp', '-O3', '-march=native'],
            extra_ldflags=['-fopenmp'],
            with_cuda=False,
            verbose=False,
        )
    except Exception:
        _ext = load_inline(
            name='ce_stage2_cpp_ext',
            cpp_sources=cpp_source,
            extra_cflags=['-fopenmp', '-O3'],
            extra_ldflags=['-fopenmp'],
            with_cuda=False,
            verbose=True,
        )
    return _ext


def ce_optimize_stage2_cpp(locs_list, z_list, X_list, knots,
                           mu0, Sigma0, beta0, delta0, theta0,
                           T=5, S=5, num_threads=8,
                           hessian_mode="analytical",
                           dtype=torch.float64):
    """Run ce_optimize_stage2 entirely in C++ with OpenMP parallelism.

    Args:
        locs_list:     list of J tensors, each (n_j, 2)
        z_list:        list of J tensors, each (n_j, 1)
        X_list:        list of J tensors, each (n_j, p) — empty tensors if p=0
        knots:         (m, 2) tensor
        mu0, Sigma0, beta0, delta0, theta0: initial parameters
        T:             outer iterations
        S:             max Newton steps per iteration
        num_threads:   OpenMP threads (= J)
        hessian_mode:  "analytical" (default, matches Python autograd), "forward_fd",
                       or "central_fd"
        dtype:         torch.float64 (default) or torch.float32. FP32 halves the
                       memory bandwidth pressure on the big (n,m) buffers; the
                       small (m,m) prior gradient/Hessian is computed in FP64
                       internally regardless for numerical stability.

    Returns:
        mu, Sigma, beta_list, delta_list, theta_list, s_list  (all in `dtype`)
    """
    ext = _get_ext()
    assert dtype in (torch.float32, torch.float64), "dtype must be float32 or float64"

    def _conv(t):
        if isinstance(t, (int, float)):
            return torch.tensor([float(t)], dtype=dtype)
        return t.to(dtype=dtype).contiguous()

    locs_vec = [_conv(t) for t in locs_list]
    z_vec    = [_conv(t) for t in z_list]
    X_vec    = [_conv(t) if (isinstance(t, torch.Tensor) and t.numel() > 0)
                else torch.empty(0, dtype=dtype) for t in X_list]
    knots_t  = _conv(knots)
    mu0_t    = _conv(mu0).reshape(-1, 1)
    Sigma0_t = _conv(Sigma0)
    beta0_t  = _conv(beta0).reshape(-1, 1)
    delta0_t = _conv(delta0).reshape(-1)
    theta0_t = _conv(theta0).reshape(-1)

    return ext.ce_optimize_stage2_cpp_impl(
        locs_vec, z_vec, X_vec, knots_t, mu0_t, Sigma0_t, beta0_t, delta0_t, theta0_t,
        T, S, num_threads, hessian_mode)
