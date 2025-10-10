import math
import numpy as np
from typing import List, Tuple, Optional, Union
from .hawkes import HawkesExponential
from workflow.features.exogenous import ExogenousDesign


class CoxHawkesExponential:
    """
    Cox×Hawkes with exponential kernel and log-link exogenous baseline:

      λ_i(t) = exp(θ_i^T X(t)) + Σ_j Σ_{t_k^j < t} α_{ij} e^{-β_{ij}(t - t_k^j)}

    Supports likelihood and residuals with piecewise-constant exogenous features.
    """

    def __init__(self, theta: np.ndarray, alpha: np.ndarray, beta: Union[np.ndarray, float], exo: ExogenousDesign):
        theta = np.asarray(theta, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        if np.isscalar(beta):
            beta = float(beta)
            beta = np.full_like(alpha, beta, dtype=float)
        else:
            beta = np.asarray(beta, dtype=float)
        if theta.ndim != 2:
            raise ValueError("theta must be (D, K)")
        d, k = theta.shape
        if alpha.shape != (d, d) or beta.shape != (d, d):
            raise ValueError("alpha/beta shape mismatch")
        if np.any(alpha < 0) or np.any(beta <= 0):
            raise ValueError("require alpha>=0, beta>0")
        self.dim = d
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.exo = exo

    def _baseline_integral_i(self, i: int) -> float:
        return self.exo.integral_exp_theta(self.theta[i])

    def _baseline_grad_i(self, i: int) -> np.ndarray:
        return self.exo.integral_exp_theta_times_X(self.theta[i])

    def loglikelihood(self, events: List[Tuple[float, int]], T: float) -> float:
        if len(events) == 0:
            base = sum(self._baseline_integral_i(i) for i in range(self.dim))
            return float(-base)
        events = sorted(events, key=lambda x: x[0])
        log_terms = 0.0
        S = np.zeros_like(self.alpha)
        t_prev = 0.0
        for t, i in events:
            dt = t - t_prev
            if dt < 0:
                raise ValueError("Events must be time-ordered")
            S = S * np.exp(-self.beta * dt)
            lam_i = math.exp(float(self.theta[i] @ self.exo.value_at(t))) + float((self.alpha[i, :] * S[i, :]).sum())
            if lam_i <= 0:
                lam_i = 1e-300
            log_terms += math.log(lam_i)
            S[:, i] += 1.0
            t_prev = t
        # Integral part
        integral = sum(self._baseline_integral_i(i) for i in range(self.dim))
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
                    if t_j >= T:
                        continue
                    x = beta_ij * (T - t_j)
                    contrib += -math.expm1(-x)
                integral += float(self.alpha[i, j] / beta_ij) * contrib
        return float(log_terms - integral)

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
            # baseline integral on (t_prev, t]: exact integral with piecewise-constant design
            base = 0.0
            for d in range(self.dim):
                base += self.exo.integral_exp_theta_between(self.theta[d], t_prev, t)
            if dt > 0:
                term = (self.alpha / self.beta) * (1.0 - np.exp(-self.beta * dt)) * S
                base += float(term.sum())
            resids.append(base)
            S = S * np.exp(-self.beta * dt)
            S[:, i] += 1.0
            t_prev = t
        return resids


