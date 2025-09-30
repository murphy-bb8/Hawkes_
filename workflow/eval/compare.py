import numpy as np
from typing import List, Tuple

from ..models.hawkes import HawkesExponential
from ..baselines import PoissonProcess, marked_events_to_times, mle_rate_from_times


def aic(loglik: float, num_params: int) -> float:
    return 2 * num_params - 2 * loglik


def compare_hawkes_poisson(events: List[Tuple[float, int]], T: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    events = sorted(events, key=lambda x: x[0])
    hawkes = HawkesExponential(mu, alpha, beta)
    ll_h = hawkes.loglikelihood(events, T)
    k_h = mu.size + alpha.size + beta.size
    aic_h = aic(ll_h, k_h)
    times = marked_events_to_times(events)
    lam_hat = mle_rate_from_times(times, T)
    pois = PoissonProcess(lam_hat)
    ll_p = pois.loglik(times, T)
    k_p = 1
    aic_p = aic(ll_p, k_p)
    return {
        'hawkes_loglik': ll_h,
        'hawkes_aic': aic_h,
        'poisson_lambda': lam_hat,
        'poisson_loglik': ll_p,
        'poisson_aic': aic_p,
    }


