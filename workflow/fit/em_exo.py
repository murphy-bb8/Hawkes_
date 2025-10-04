from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class EMExoResult:
    alpha: np.ndarray
    beta: np.ndarray
    n_iter: int


def em_update_alpha(events: List[Tuple[float, int]], T: float, dim: int, theta: np.ndarray,
                    alpha: np.ndarray, beta: np.ndarray, max_iter: int = 50) -> EMExoResult:
    """
    Simple EM-like updates for alpha with fixed theta, beta.
    E-step: responsibilities z_k^{i<-j} via fraction of intensity from excitation over total.
    M-step: alpha_{ij} <- sum z / sum ∫ e^{-beta (t - s)} dN_j(s)
    """
    events_sorted = sorted(events, key=lambda x: x[0])
    times_by_type = [[] for _ in range(dim)]
    for t, i in events_sorted:
        times_by_type[i].append(t)

    for it in range(1, max_iter + 1):
        S = np.zeros_like(alpha)
        z_acc = np.zeros_like(alpha)
        denom = np.zeros_like(alpha)
        t_prev = 0.0
        for t, i in events_sorted:
            dt = t - t_prev
            decay = np.exp(-beta * dt)
            S = S * decay
            # intensities
            base_i = 1.0  # use proportional baseline for z only (theta absorbed)
            excite_vec = alpha[i, :] * S[i, :]
            lam_i = base_i + float(excite_vec.sum())
            if lam_i <= 0:
                lam_i = 1e-300
            # responsibilities to each source j
            if excite_vec.sum() > 0:
                z_ij = (excite_vec / lam_i)
                z_acc[i, :] += z_ij
            S[:, i] += 1.0
            t_prev = t
        # denominator ∑ ∫ e^{-β (T - s)} dN_j(s)
        for i in range(dim):
            for j in range(dim):
                if times_by_type[j]:
                    t_arr = np.asarray(times_by_type[j], dtype=float)
                    denom[i, j] = np.sum(1.0 - np.exp(-beta[i, j] * (T - t_arr))) / beta[i, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            new_alpha = np.where(denom > 0, z_acc / denom, alpha)
        alpha = np.clip(new_alpha, 0.0, 1e6)
    return EMExoResult(alpha=alpha, beta=beta, n_iter=max_iter)


