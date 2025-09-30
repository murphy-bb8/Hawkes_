import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..models.hawkes import HawkesExponential
from numpy.linalg import eigvals


@dataclass
class FitResult:
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    converged: bool
    n_iter: int


def _project_nonneg(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x[x < eps] = eps
    return x


def fit_hawkes_exponential(
    events: List[Tuple[float, int]],
    T: float,
    dim: int,
    init_mu: Optional[np.ndarray] = None,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    max_iter: int = 300,
    step_mu: float = 1e-2,
    step_alpha: float = 1e-2,
    step_beta: float = 5e-4,
    tol: float = 1e-6,
    seed: Optional[int] = None,
    min_beta: float = 0.1,
    l2_alpha: float = 0.0,
    rho_max: float = 0.95,
) -> FitResult:
    rng = np.random.default_rng(seed)
    events = sorted(events, key=lambda x: x[0])

    if init_mu is None:
        rate = len(events) / max(T, 1e-9)
        init_mu = np.full(dim, max(1e-3, 0.5 * rate / max(dim, 1)))
    if init_alpha is None:
        init_alpha = np.full((dim, dim), 0.1)
    if init_beta is None:
        init_beta = np.full((dim, dim), 1.0)

    mu = _project_nonneg(np.asarray(init_mu, dtype=float))
    alpha = _project_nonneg(np.asarray(init_alpha, dtype=float))
    beta = _project_nonneg(np.asarray(init_beta, dtype=float))
    beta[beta < min_beta] = min_beta

    # Project to spectral radius constraint if needed
    def _enforce_rho(alpha_mat: np.ndarray, beta_mat: np.ndarray) -> np.ndarray:
        G = alpha_mat / beta_mat
        rho = float(max(abs(eigvals(G))))
        if rho_max is not None and rho_max > 0 and rho > rho_max:
            scale = rho_max / rho
            return alpha_mat * scale
        return alpha_mat

    alpha = _enforce_rho(alpha, beta)

    model = HawkesExponential(mu, alpha, beta)
    best_ll = model.loglikelihood(events, T)
    best_params = (mu.copy(), alpha.copy(), beta.copy())

    times_by_type: List[List[float]] = [[] for _ in range(dim)]
    for t, i in events:
        times_by_type[i].append(t)

    M = np.zeros((dim, dim))

    for it in range(1, max_iter + 1):
        S = np.zeros((dim, dim))
        t_prev = 0.0
        g_mu = np.zeros(dim)
        g_alpha = np.zeros((dim, dim))
        g_beta = np.zeros((dim, dim))

        for (t, i_mark) in events:
            dt = t - t_prev
            if dt < 0:
                raise ValueError("Events must be sorted by time")
            decay = np.exp(-beta * dt)
            S = S * decay
            M = M * decay + S * dt
            lam_i = mu[i_mark] + float((alpha[i_mark, :] * S[i_mark, :]).sum())
            lam_i = max(lam_i, 1e-12)
            g_mu[i_mark] += 1.0 / lam_i
            g_alpha[i_mark, :] += S[i_mark, :] / lam_i
            g_beta[i_mark, :] += -alpha[i_mark, :] * (M[i_mark, :]) / lam_i
            S[:, i_mark] += 1.0
            t_prev = t

        g_mu -= T

        for i in range(dim):
            for j in range(dim):
                if alpha[i, j] <= 0 and g_alpha[i, j] == 0 and g_beta[i, j] == 0:
                    continue
                b = beta[i, j]
                sum_one_minus = 0.0
                sum_T_minus_exp = 0.0
                for t_j in times_by_type[j]:
                    dtT = T - t_j
                    e = math.exp(-b * dtT)
                    sum_one_minus += (1.0 - e)
                    sum_T_minus_exp += dtT * e
                g_alpha[i, j] += -(1.0 / b) * sum_one_minus
                g_beta[i, j] += -alpha[i, j] * ( -sum_one_minus / (b * b) + sum_T_minus_exp / b )

        # L2 正则到 alpha（抑制过大分枝比）
        if l2_alpha > 0:
            g_alpha -= 2.0 * l2_alpha * alpha

        mu = _project_nonneg(mu + step_mu * g_mu)
        alpha = _project_nonneg(alpha + step_alpha * g_alpha)
        beta = _project_nonneg(beta + step_beta * g_beta)
        beta[beta < min_beta] = min_beta

        # spectral radius projection on alpha
        alpha = _enforce_rho(alpha, beta)

        model = HawkesExponential(mu, alpha, beta)
        ll = model.loglikelihood(events, T)
        if ll > best_ll + tol:
            best_ll = ll
            best_params = (mu.copy(), alpha.copy(), beta.copy())

        if np.linalg.norm(step_mu * g_mu) + np.linalg.norm(step_alpha * g_alpha) + np.linalg.norm(step_beta * g_beta) < tol:
            mu, alpha, beta = best_params
            return FitResult(mu, alpha, beta, best_ll, True, it)

    mu, alpha, beta = best_params
    return FitResult(mu, alpha, beta, best_ll, False, max_iter)


