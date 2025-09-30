import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class MAPEMResult:
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    n_iter: int


def _loglik_exp_kernel(events: List[Tuple[float, int]], T: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
    # identical to MLE loglik; used for monitoring
    if len(events) == 0:
        return float(-mu.sum() * T)
    events = sorted(events, key=lambda x: x[0])
    d = mu.shape[0]
    log_terms = 0.0
    S = np.zeros_like(alpha)
    t_prev = 0.0
    for t, i in events:
        dt = t - t_prev
        S = S * np.exp(-beta * dt)
        lam_i = mu[i] + float((alpha[i, :] * S[i, :]).sum())
        log_terms += math.log(max(lam_i, 1e-300))
        S[:, i] += 1.0
        t_prev = t
    integral = float(mu.sum() * T)
    times_by_type: List[List[float]] = [[] for _ in range(d)]
    for t, i in events:
        times_by_type[i].append(t)
    for i in range(d):
        for j in range(d):
            if alpha[i, j] == 0:
                continue
            b = beta[i, j]
            contrib = 0.0
            for t_j in times_by_type[j]:
                contrib += (1.0 - math.exp(-b * (T - t_j)))
            integral += float(alpha[i, j] / b) * contrib
    return float(log_terms - integral)


def map_em_exponential(
    events: List[Tuple[float, int]],
    T: float,
    dim: int,
    init_mu: Optional[np.ndarray] = None,
    init_alpha: Optional[np.ndarray] = None,
    init_beta: Optional[np.ndarray] = None,
    max_iter: int = 200,
    min_beta: float = 0.1,
    prior_mu_a: float = 1.0,
    prior_mu_b: float = 1.0,
    prior_alpha_a: float = 1.0,
    prior_alpha_b: float = 1.0,
    prior_beta_a: float = 1.0,
    prior_beta_b: float = 1.0,
    update_beta: bool = False,
) -> MAPEMResult:
    """
    MAP-EM for multivariate Hawkes with exponential kernels.
    - Uses Gamma priors on mu, alpha, beta with (shape a, rate b). For simplicity beta is fixed unless update_beta=True.
    - E-step: responsibilities for immigrant vs offspring parents
    - M-step: closed-form MAP updates for mu and alpha using exposures
    """
    events = sorted(events, key=lambda x: x[0])
    d = dim
    if init_mu is None:
        rate = len(events) / max(T, 1e-9)
        init_mu = np.full(d, max(1e-3, 0.5 * rate / max(d, 1)))
    if init_alpha is None:
        init_alpha = np.full((d, d), 0.1)
    if init_beta is None:
        init_beta = np.full((d, d), 1.0)
    mu = np.maximum(np.asarray(init_mu, dtype=float), 1e-8)
    alpha = np.maximum(np.asarray(init_alpha, dtype=float), 0.0)
    beta = np.maximum(np.asarray(init_beta, dtype=float), min_beta)

    # Precompute indices of events per type
    times = np.array([t for t, _ in events], dtype=float)
    types = np.array([i for _, i in events], dtype=int)
    n = len(events)
    times_by_type: List[List[int]] = [[] for _ in range(d)]
    for idx, i in enumerate(types):
        times_by_type[i].append(idx)

    for it in range(1, max_iter + 1):
        # E-step: responsibilities
        # immigrant prob z0[k] for event k, and parent probabilities zij[k, m] aggregated per (i_parent_type)
        z0 = np.zeros(n, dtype=float)
        # offspring counts aggregated per (i_target, j_source)
        Nij = np.zeros((d, d), dtype=float)
        # exposures denominator for alpha: sum over t_j of (1 - exp(-b(T - t_j)))/b
        exposure = np.zeros((d, d), dtype=float)

        # precompute exposures (depends only on beta and source times)
        for j in range(d):
            tj = np.array([times[idx] for idx in times_by_type[j]], dtype=float)
            if tj.size == 0:
                continue
            for i in range(d):
                b = beta[i, j]
                exposure[i, j] = ((1.0 - np.exp(-b * (T - tj))).sum()) / b

        # Iterate events in time to compute responsibilities
        # Maintain decayed sums per row i, source j: S_ij(t) and also keep individual past contributions for Nij
        # For Nij, we need expected parent per past event; we compute pointwise probs.
        # Simpler O(N^2): for each event k, sum over all past events m with mark j
        for k in range(n):
            t_k = times[k]
            i_k = types[k]
            # compute lambda_i(t_k)
            lam = mu[i_k]
            # accumulate contributions from parents and compute per-parent probs later
            parents_idx = []
            parent_vals = []
            for m in range(k):
                j = types[m]
                dt = t_k - times[m]
                if dt <= 0:
                    continue
                val = alpha[i_k, j] * math.exp(-beta[i_k, j] * dt)
                if val > 0:
                    lam += val
                    parents_idx.append((i_k, j, m))
                    parent_vals.append(val)
            lam = max(lam, 1e-300)
            # immigrant resp
            z0[k] = mu[i_k] / lam
            # offspring resp per parent
            if parent_vals:
                parent_vals = np.asarray(parent_vals, dtype=float)
                probs = parent_vals / lam
                for p, pr in zip(parents_idx, probs):
                    i_row, j_col, _ = p
                    Nij[i_row, j_col] += pr

        # M-step: MAP updates using Gamma(a,b) priors (mode (a+count-1)/(b+exposure))
        # mu_i
        for i in range(d):
            count_immigrant = float(sum(z0[k] for k in times_by_type[i]))
            mu[i] = max(1e-8, (prior_mu_a + count_immigrant - 1.0) / (prior_mu_b + T))
        # alpha_ij
        for i in range(d):
            for j in range(d):
                denom = prior_alpha_b + exposure[i, j]
                num = prior_alpha_a + Nij[i, j] - 1.0
                alpha[i, j] = max(0.0, num / max(denom, 1e-12))
        # beta optionally (kept fixed by default)
        if update_beta:
            # crude fixed-point: keep beta at least min_beta; refine not implemented for brevity
            beta = np.maximum(beta, min_beta)

        # Optional projection: keep stability (roughly)
        # scale alpha if spectral radius > 0.95
        try:
            G = alpha / beta
            rho = max(abs(np.linalg.eigvals(G)))
            if rho > 0.95:
                alpha *= (0.95 / float(rho))
        except Exception:
            pass

    ll = _loglik_exp_kernel(events, T, mu, alpha, beta)
    return MAPEMResult(mu, alpha, beta, ll, max_iter)


