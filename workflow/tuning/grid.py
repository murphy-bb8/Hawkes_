import itertools
import numpy as np
from typing import Dict, Any, List, Tuple

from ..fit import fit_hawkes_exponential
from ..models.hawkes import HawkesExponential
from ..eval import compare_hawkes_poisson


def _residual_score(model: HawkesExponential, events: List[Tuple[float, int]], T: float) -> float:
    """
    期望 Exp(1)：均值≈1、方差≈1。用 |mean-1| + 0.25*|var-1| 作为惩罚。
    若无残差（极少见），返回中等惩罚值以避免错误排序。
    """
    res = model.compensate_residuals(events, T)
    if len(res) == 0:
        return 1.0
    arr = np.asarray(res, dtype=float)
    mean_err = abs(arr.mean() - 1.0)
    var_err = abs(arr.var() - 1.0)
    return float(mean_err + 0.25 * var_err)


def grid_search(
    events: List[Tuple[float, int]],
    T: float,
    dim: int,
    grid_min_beta: List[float],
    grid_l2_alpha: List[float],
    grid_rho_max: List[float],
    max_iter: int = 1500,
    step_mu: float = 1e-2,
    step_alpha: float = 1e-2,
    step_beta: float = 2e-4,
) -> Dict[str, Any]:
    combos = list(itertools.product(grid_min_beta, grid_l2_alpha, grid_rho_max))
    results = []
    best = None
    best_score = float('inf')
    for (min_beta, l2_alpha, rho_max) in combos:
        fit = fit_hawkes_exponential(
            events,
            T=T,
            dim=dim,
            max_iter=max_iter,
            step_mu=step_mu,
            step_alpha=step_alpha,
            step_beta=step_beta,
            min_beta=min_beta,
            l2_alpha=l2_alpha,
            rho_max=rho_max,
        )
        model = HawkesExponential(fit.mu, fit.alpha, fit.beta)
        comp = compare_hawkes_poisson(events, T, fit.mu, fit.alpha, fit.beta)
        res_pen = _residual_score(model, events, T)
        score = comp['hawkes_aic'] + 5.0 * res_pen
        record = {
            'min_beta': float(min_beta),
            'l2_alpha': float(l2_alpha),
            'rho_max': float(rho_max),
            'hawkes_aic': float(comp['hawkes_aic']),
            'poisson_aic': float(comp['poisson_aic']),
            'residual_penalty': float(res_pen),
            'score': float(score),
            'fit_mu': fit.mu.tolist(),
            'fit_alpha': fit.alpha.tolist(),
            'fit_beta': fit.beta.tolist(),
            'converged': fit.converged,
            'iters': fit.n_iter,
        }
        results.append(record)
        if score < best_score:
            best_score = score
            best = record
    return {'best': best, 'results': results}

