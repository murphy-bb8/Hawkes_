import numpy as np
from typing import Dict, Any, List
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def ks_exp_test(residuals: List[float]) -> Dict[str, Any]:
    arr = np.asarray(residuals, dtype=float)
    if arr.size == 0:
        return {'D': None, 'p_value': None, 'n': 0}
    # Under Exp(1), residuals are exponential with mean=1
    # Transform to uniform: U = 1 - exp(-r)
    u = 1.0 - np.exp(-arr)
    d, p = stats.kstest(arr, 'expon', args=(0, 1.0))
    return {'D': float(d), 'p_value': float(p), 'n': int(arr.size), 'U': u}


def compute_uniform_from_residuals(residuals: List[float]) -> np.ndarray:
    arr = np.asarray(residuals, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return 1.0 - np.exp(-arr)


def ks_uniform_test(u: np.ndarray) -> Dict[str, Any]:
    if u.size == 0:
        return {'D': None, 'p_value': None, 'n': 0}
    d, p = stats.kstest(u, 'uniform', args=(0, 1))
    return {'D': float(d), 'p_value': float(p), 'n': int(u.size)}


def ljung_box_test(u: np.ndarray, lags: int = 20) -> Dict[str, Any]:
    if u.size == 0:
        return {'Q': None, 'p_value': None, 'n': 0}
    # Remove mean (for autocorrelation) and compute LB test
    x = u - u.mean()
    lb = acorr_ljungbox(x, lags=[lags], return_df=True)
    Q = float(lb['lb_stat'].iloc[0])
    p = float(lb['lb_pvalue'].iloc[0])
    return {'Q': Q, 'p_value': p, 'n': int(u.size), 'lags': lags}


