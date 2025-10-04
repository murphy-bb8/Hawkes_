import math
import numpy as np
from typing import List, Tuple, Optional, Union


class HawkesExponential:
    """
    Multivariate Hawkes process with exponential kernels.

    Intensity for dimension i:
        lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_{ij} * exp(-beta_{ij} * (t - t_k^j))

    Parameters
    ----------
    mu : (D,) array_like
        Baseline intensity per dimension.
    alpha : (D, D) array_like
        Excitation matrix, alpha_ij >= 0.
    beta : float or (D, D) array_like
        Decay rate(s) for exponential kernel. beta > 0.
    """

    def __init__(self, mu: np.ndarray, alpha: np.ndarray, beta: Union[np.ndarray, float]):
        mu = np.asarray(mu, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        if np.isscalar(beta):
            beta = float(beta)
            beta = np.full_like(alpha, beta, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)

        if mu.ndim != 1:
            raise ValueError("mu must be 1-D array")
        d = mu.shape[0]
        if alpha.shape != (d, d):
            raise ValueError("alpha must be shape (D, D)")
        if beta.shape != (d, d):
            raise ValueError("beta must be shape (D, D)")
        if np.any(mu < 0) or np.any(alpha < 0) or np.any(beta <= 0):
            raise ValueError("Invalid parameters: require mu>=0, alpha>=0, beta>0")

        self.dim = d
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def simulate_ogata(self, T: float, max_jumps: int = 1_000_000, seed: Optional[int] = None) -> List[Tuple[float, int]]:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        events: List[Tuple[float, int]] = []
        t = 0.0
        S = np.zeros_like(self.alpha)
        # Precompute conservative upper bound per dim: mu_i + sum_j alpha_ij * N_max
        # For Bund-scale simulation, adaptively refresh lam_bar using decayed S to tighten bound.
        while t < T and len(events) < max_jumps:
            lam_vec = self.mu + (self.alpha * S).sum(axis=1)
            lam_vec = np.clip(lam_vec, 0.0, np.inf)
            lam_bar = float(lam_vec.sum())
            if lam_bar <= 0:
                break
            w = rng.exponential(1.0 / lam_bar)
            t_candidate = t + w
            if t_candidate > T:
                break
            dt = t_candidate - t
            decay = np.exp(-self.beta * dt)
            S = S * decay
            lam_vec = self.mu + (self.alpha * S).sum(axis=1)
            lam_vec = np.clip(lam_vec, 0.0, np.inf)
            lam_sum = float(lam_vec.sum())
            if rng.uniform() <= (lam_sum / lam_bar if lam_bar > 0 else 0.0):
                if lam_sum <= 0:
                    t = t_candidate
                    continue
                probs = lam_vec / lam_sum
                i = int(rng.choice(self.dim, p=probs))
                events.append((t_candidate, i))
                S[:, i] += 1.0
                t = t_candidate
            else:
                t = t_candidate
        return events

    def loglikelihood(self, events: List[Tuple[float, int]], T: float) -> float:
        if len(events) == 0:
            return float(-self.mu.sum() * T)
        events = sorted(events, key=lambda x: x[0])
        log_terms = 0.0
        S = np.zeros_like(self.alpha)
        t_prev = 0.0
        for t, i in events:
            dt = t - t_prev
            if dt < 0:
                raise ValueError("Events must be time-ordered")
            S = S * np.exp(-self.beta * dt)
            lam_i = self.mu[i] + float((self.alpha[i, :] * S[i, :]).sum())
            if lam_i <= 0:
                lam_i = 1e-300
            log_terms += math.log(lam_i)
            S[:, i] += 1.0
            t_prev = t
        integral = float(self.mu.sum() * T)
        times_by_type: List[List[float]] = [[] for _ in range(self.dim)]
        for t, i in events:
            times_by_type[i].append(t)
        for i in range(self.dim):
            for j in range(self.dim):
                if self.alpha[i, j] == 0:
                    continue
                beta_ij = self.beta[i, j]
                contrib = 0.0
                for t_j in times_by_type[j]:
                    contrib += (1.0 - math.exp(-beta_ij * (T - t_j)))
                integral += float(self.alpha[i, j] / beta_ij) * contrib
        return float(log_terms - integral)

    def branching_ratio(self) -> float:
        G = self.alpha / self.beta
        eigvals = np.linalg.eigvals(G)
        return float(max(abs(eigvals)))

    def compensate_residuals(self, events: List[Tuple[float, int]], T: float) -> List[float]:
        if len(events) == 0:
            return []
        events = sorted(events, key=lambda x: x[0])
        resids: List[float] = []
        S = np.zeros_like(self.alpha)
        t_prev = 0.0
        for t, i in events:
            dt = t - t_prev
            if dt < 0:
                raise ValueError("Events must be ordered by time")
            integ = float(self.mu.sum()) * dt
            if dt > 0:
                term = (self.alpha / self.beta) * (1.0 - np.exp(-self.beta * dt)) * S
                integ += float(term.sum())
            resids.append(integ)
            S = S * np.exp(-self.beta * dt)
            S[:, i] += 1.0
            t_prev = t
        return resids

    def intensity_over_grid(self, events: List[Tuple[float, int]], grid: np.ndarray) -> np.ndarray:
        d = self.dim
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("grid must be 1-D")
        events = sorted(events, key=lambda x: x[0])
        intensities = np.zeros((grid.shape[0], d), dtype=float)
        S = np.zeros_like(self.alpha)
        t_prev = 0.0
        k = 0
        for idx, t in enumerate(grid):
            dt = t - t_prev
            if dt < 0:
                raise ValueError("grid must be ascending")
            S = S * np.exp(-self.beta * dt)
            while k < len(events) and events[k][0] <= t:
                _, j = events[k]
                S[:, j] += 1.0
                k += 1
            lam_vec = self.mu + (self.alpha * S).sum(axis=1)
            intensities[idx, :] = np.clip(lam_vec, 0.0, np.inf)
            t_prev = t
        return intensities


def convert_univariate_times_to_marked(times: List[float]) -> List[Tuple[float, int]]:
    return [(float(t), 0) for t in times]


