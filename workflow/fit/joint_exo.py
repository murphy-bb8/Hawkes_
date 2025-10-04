from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from workflow.features.exogenous import ExogenousDesign


@dataclass
class JointExoResult:
    theta: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    n_iter: int


def _spectral_radius(matrix: np.ndarray) -> float:
    # Guard against NaNs/Infs
    if not np.all(np.isfinite(matrix)):
        return float('inf')
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


def fit_cox_hawkes_joint(
    events: List[Tuple[float, int]],
    T: float,
    dim: int,
    exo: ExogenousDesign,
    init_theta: Optional[np.ndarray] = None,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    max_iter: int = 500,
    step_theta: float = 1e-3,
    step_alpha: float = 1e-3,
    step_beta: float = 1e-4,
    min_beta: float = 0.1,
    l2_alpha: float = 0.0,
    rho_max: float = 0.95,
) -> JointExoResult:
    """
    Block-free joint gradient ascent for Cox×Hawkes parameters (θ, α, β).

    Objective: log-likelihood with optional L2(α) penalty; constraints handled by
    projection (α>=0, β>=min_beta, spectral radius(α/β)<=rho_max).
    """
    if len(events) == 0:
        raise ValueError("events must be non-empty for joint fitting")
    events_sorted = sorted(events, key=lambda x: x[0])
    # Initialize
    K = exo.num_features
    theta = np.zeros((dim, K), dtype=float) if init_theta is None else init_theta.copy()
    alpha = np.full((dim, dim), 0.1, dtype=float) if init_alpha is None else init_alpha.copy()
    beta = np.full((dim, dim), 1.0, dtype=float) if init_beta is None else init_beta.copy()
    beta = np.maximum(beta, min_beta)

    # Precompute source times per dimension for integral terms
    times_by_type: List[List[float]] = [[] for _ in range(dim)]
    for t, i in events_sorted:
        if 0.0 <= t <= T:
            times_by_type[i].append(t)

    m_th = np.zeros_like(theta); v_th = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, max_iter + 1):
        # State variables for exponential kernels
        S = np.zeros_like(alpha)  # decayed counts per (i <- j)
        R = np.zeros_like(alpha)  # age-weighted decayed counts per (i <- j)

        grad_theta = np.zeros_like(theta)
        grad_alpha = np.zeros_like(alpha)
        grad_beta = np.zeros_like(beta)

        t_prev = 0.0
        for t, i in events_sorted:
            dt = t - t_prev
            if dt < 0:
                raise ValueError("events must be ordered by time")
            # decay both S and R to time t
            decay = np.exp(-beta * dt)
            R = R * decay + dt * decay * S
            S = S * decay

            # intensity for mark i
            x_t = exo.value_at(t)
            base_i = np.exp(float(theta[i] @ x_t))
            excite_i = float((alpha[i, :] * S[i, :]).sum())
            lam_i = max(base_i + excite_i, 1e-300)

            # theta gradient (only for the occurred dimension)
            grad_theta[i] += (base_i / lam_i) * x_t

            # alpha and beta event gradients
            # d/d alpha[i,j]: S[i,j] / lam_i
            grad_alpha[i, :] += S[i, :] / lam_i
            # d/d beta[i,j]: (alpha[i,j]/lam_i) * ( - R[i,j] )
            grad_beta[i, :] += (alpha[i, :] / lam_i) * (-R[i, :])

            # incorporate the new event for all targets from source i
            S[:, i] += 1.0
            t_prev = t

        # subtract integral contributions
        # theta integrals: - ∫ exp(theta_i^T X(t)) X(t) dt
        for i in range(dim):
            grad_theta[i] -= exo.integral_exp_theta_times_X(theta[i])

        # alpha/beta integrals over [0, T]
        for i in range(dim):
            for j in range(dim):
                if times_by_type[j]:
                    t_arr = np.asarray(times_by_type[j], dtype=float)
                    delta = T - t_arr
                    e = np.exp(-beta[i, j] * delta)
                    sum1 = np.sum(1.0 - e)
                    sum2 = np.sum(delta * e)
                    # alpha integral: -(1/beta) * sum1
                    grad_alpha[i, j] -= (1.0 / beta[i, j]) * sum1
                    # beta integral: -alpha * ( -sum1/beta^2 + sum2/beta )
                    grad_beta[i, j] -= alpha[i, j] * (-(sum1) / (beta[i, j] ** 2) + (sum2) / beta[i, j])

        # L2 regularization on alpha (maximize loglik - 0.5*l2*||alpha||^2)
        if l2_alpha > 0.0:
            grad_alpha -= l2_alpha * alpha

        # Parameter updates with clipping to maintain finiteness
        # Adam for theta
        m_th = b1 * m_th + (1 - b1) * grad_theta
        v_th = b2 * v_th + (1 - b2) * (grad_theta * grad_theta)
        mh = m_th / (1 - b1 ** it)
        vh = v_th / (1 - b2 ** it)
        theta += step_theta * mh / (np.sqrt(vh) + eps)
        theta = np.clip(theta, -20.0, 20.0)
        alpha = np.maximum(0.0, alpha + step_alpha * grad_alpha)
        alpha = np.clip(alpha, 0.0, 1e6)
        beta = np.maximum(min_beta, beta + step_beta * grad_beta)
        beta = np.clip(beta, min_beta, 1e6)

        # Spectral radius projection for stability; handle non-finite via fallback
        G = alpha / beta
        if not np.all(np.isfinite(G)):
            rho = float('inf')
        else:
            rho = _spectral_radius(G)
        if rho > rho_max and rho > 0:
            alpha *= (rho_max / rho)

    # Compute final log-likelihood
    # We reuse the structure from models/cox_hawkes if available, but avoid circular import.
    # A light evaluation here:
    # event term
    S = np.zeros_like(alpha)
    log_terms = 0.0
    t_prev = 0.0
    for t, i in events_sorted:
        dt = t - t_prev
        S = S * np.exp(-beta * dt)
        x_t = exo.value_at(t)
        base_i = np.exp(float(theta[i] @ x_t))
        excite_i = float((alpha[i, :] * S[i, :]).sum())
        lam_i = max(base_i + excite_i, 1e-300)
        log_terms += float(np.log(lam_i))
        S[:, i] += 1.0
        t_prev = t
    # integral term
    integral = 0.0
    for i in range(dim):
        integral += exo.integral_exp_theta(theta[i])
    for i in range(dim):
        for j in range(dim):
            if times_by_type[j]:
                t_arr = np.asarray(times_by_type[j], dtype=float)
                integral += float(alpha[i, j] / beta[i, j]) * np.sum(1.0 - np.exp(-beta[i, j] * (T - t_arr)))

    ll = float(log_terms - integral)
    return JointExoResult(theta=theta, alpha=alpha, beta=beta, loglik=ll, n_iter=max_iter)


