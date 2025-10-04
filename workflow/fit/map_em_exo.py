from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from workflow.features.exogenous import ExogenousDesign


@dataclass
class MAPEMExoResult:
    theta: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    n_iter: int


def map_em_exogenous(
    events: List[Tuple[float, int]],
    T: float,
    dim: int,
    exo: ExogenousDesign,
    init_theta: Optional[np.ndarray] = None,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    max_iter: int = 200,
    step_theta: float = 1e-3,
    min_beta: float = 0.4,
    rho_max: float = 0.9,
    prior_alpha_a: float = 1.0,
    prior_alpha_b: float = 1.0,
) -> MAPEMExoResult:
    events_sorted = sorted(events, key=lambda x: x[0])
    K = exo.num_features
    theta = np.zeros((dim, K), dtype=float) if init_theta is None else init_theta.copy()
    alpha = np.full((dim, dim), 0.1, dtype=float) if init_alpha is None else init_alpha.copy()
    beta = np.full((dim, dim), 1.0, dtype=float) if init_beta is None else init_beta.copy()
    beta = np.maximum(beta, min_beta)

    def spectral_radius(A: np.ndarray) -> float:
        if not np.all(np.isfinite(A)):
            return float('inf')
        vals = np.linalg.eigvals(A)
        return float(np.max(np.abs(vals)))

    for it in range(1, max_iter + 1):
        # E-step: responsibilities
        S = np.zeros_like(alpha)
        R = np.zeros_like(alpha)
        z_acc = np.zeros_like(alpha)
        t_prev = 0.0
        for t, i in events_sorted:
            dt = t - t_prev
            decay = np.exp(-beta * dt)
            R = R * decay + dt * decay * S
            S = S * decay
            # intensities
            x_t = exo.value_at(t)
            base_i = np.exp(float(theta[i] @ x_t))
            excite_vec = alpha[i, :] * S[i, :]
            lam_i = base_i + float(excite_vec.sum())
            if lam_i <= 0:
                lam_i = 1e-300
            # responsibility to each j
            if excite_vec.sum() > 0:
                z_acc[i, :] += (excite_vec / lam_i)
            S[:, i] += 1.0
            t_prev = t

        # M-step: update alpha with Gamma prior (a,b)
        # denom: ∫ e^{-β (t - s)} dN_j(s) dt over [0, T] equals sum_j (1 - e^{-β (T - t_k^j)})/β
        denom = np.zeros_like(alpha)
        times_by_type = [[] for _ in range(dim)]
        for t, i in events_sorted:
            times_by_type[i].append(t)
        for i in range(dim):
            for j in range(dim):
                if times_by_type[j]:
                    arr = np.asarray(times_by_type[j], dtype=float)
                    denom[i, j] = np.sum(1.0 - np.exp(-beta[i, j] * (T - arr))) / beta[i, j]
        alpha = (z_acc + (prior_alpha_a - 1.0)) / np.maximum(denom + prior_alpha_b, 1e-12)
        alpha = np.clip(alpha, 0.0, 1e6)

        # Update theta by one gradient step on NLL (no prior here;可扩展Gamma)
        grad_th = np.zeros_like(theta)
        # event contributions
        for t, i in events_sorted:
            x_t = exo.value_at(t)
            grad_th[i] += x_t
        # integral contributions
        for i in range(dim):
            grad_th[i] -= exo.integral_exp_theta_times_X(theta[i])
        theta += step_theta * grad_th
        theta = np.clip(theta, -20.0, 20.0)

        # Stability projection
        G = alpha / beta
        rho = spectral_radius(G)
        if rho > rho_max and rho > 0:
            alpha *= (rho_max / rho)

    # Compute loglik
    S = np.zeros_like(alpha)
    t_prev = 0.0
    log_terms = 0.0
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
    integral = 0.0
    for i in range(dim):
        integral += exo.integral_exp_theta(theta[i])
    for i in range(dim):
        for j in range(dim):
            if times_by_type[j]:
                arr = np.asarray(times_by_type[j], dtype=float)
                integral += float(alpha[i, j] / beta[i, j]) * np.sum(1.0 - np.exp(-beta[i, j] * (T - arr)))
    ll = float(log_terms - integral)
    return MAPEMExoResult(theta=theta, alpha=alpha, beta=beta, loglik=ll, n_iter=max_iter)


