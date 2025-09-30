import numpy as np
from typing import List, Tuple


def estimate_piecewise_mu(events: List[Tuple[float, int]], T: float, dim: int, bins: int = 10) -> np.ndarray:
    """
    简易分段常数基线估计：按时间将 [0,T] 划分为 bins 段，估计每段的到达率，
    最后取各段平均作为 mu 初值（多维时按类型计数/总时间）。
    这是一个粗略估计，目的是吸收季节性，提供更稳初值。
    """
    edges = np.linspace(0.0, T, bins + 1)
    counts = np.zeros((dim, bins), dtype=float)
    for t, i in events:
        if t < 0 or t >= T:
            continue
        b = int(np.searchsorted(edges, t, side='right') - 1)
        b = min(max(b, 0), bins - 1)
        counts[i, b] += 1.0
    # 段长度
    seg_T = T / bins if bins > 0 else T
    rates = counts / max(seg_T, 1e-12)
    mu_est = rates.mean(axis=1)
    mu_est = np.maximum(mu_est, 1e-6)
    return mu_est


