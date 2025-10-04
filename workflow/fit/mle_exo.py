from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from workflow.features.exogenous import ExogenousDesign


@dataclass
class CoxHawkesMLEOpt:
    theta: np.ndarray
    loglik: float
    n_iter: int


def fit_cox_hawkes_theta(events: List[Tuple[float, int]], T: float, dim: int, exo: ExogenousDesign,
                         alpha: np.ndarray, beta: np.ndarray, init_theta: Optional[np.ndarray] = None,
                         step: float = 1e-3, max_iter: int = 500, adam: bool = True,
                         grad_clip: float = 10.0, lr_decay: float = 0.0) -> CoxHawkesMLEOpt:
    if init_theta is None:
        init_theta = np.zeros((dim, exo.num_features), dtype=float)
    theta = init_theta.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    # Precompute per-dimension feature sums at event times
    events_sorted = sorted(events, key=lambda x: x[0])
    X_at_events = [exo.value_at(t) for t, _ in events_sorted]

    for it in range(1, max_iter + 1):
        grad = np.zeros_like(theta)
        # event contributions: sum over events of X(t_e) for the occurred dimension i
        for (t, i), x in zip(events_sorted, X_at_events):
            grad[i] += x
        # integral contributions: subtract ∫_0^T exp(theta_i^T X(t)) X(t) dt
        for i in range(dim):
            grad[i] -= exo.integral_exp_theta_times_X(theta[i])
        if grad_clip is not None and grad_clip > 0:
            gnorm = np.linalg.norm(grad)
            if np.isfinite(gnorm) and gnorm > grad_clip:
                grad *= (grad_clip / gnorm)
        if adam:
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * (grad * grad)
            mh = m / (1 - b1 ** it)
            vh = v / (1 - b2 ** it)
            theta += step * mh / (np.sqrt(vh) + eps)
        else:
            theta += step * grad
        if lr_decay and lr_decay > 0:
            step *= (1.0 / (1.0 + lr_decay * it))
    # simple log-likelihood proxy (without excitation integral here; focus on theta part)
    ll = 0.0
    for (t, i), x in zip(events_sorted, X_at_events):
        ll += float(theta[i] @ x)
    for i in range(dim):
        ll -= exo.integral_exp_theta(theta[i])
    return CoxHawkesMLEOpt(theta=theta, loglik=ll, n_iter=max_iter)
