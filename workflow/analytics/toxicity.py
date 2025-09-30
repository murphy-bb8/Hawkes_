import numpy as np
from typing import List, Tuple

from ..models.hawkes import HawkesExponential


def toxic_order_score(
    model: HawkesExponential,
    pre_events: List[Tuple[float, int]],
    order_time: float,
    order_mark: int,
    horizon: float = 1.0,
) -> float:
    events = sorted([e for e in pre_events if e[0] <= order_time], key=lambda x: x[0])
    S = np.zeros_like(model.alpha)
    t_prev = 0.0
    for t, j in events:
        dt = t - t_prev
        S = S * np.exp(-model.beta * dt)
        S[:, j] += 1.0
        t_prev = t
    S[:, order_mark] += 1.0
    dt_grid = np.linspace(0.0, horizon, num=64)
    inc = 0.0
    for dt in dt_grid[1:]:
        inc += float((model.alpha[:, order_mark] * np.exp(-model.beta[:, order_mark] * dt)).sum())
    inc *= horizon / (len(dt_grid) - 1)
    return inc


def predict_price_impact(
    order_flow_model: HawkesExponential,
    order_time: float,
    order_mark: int,
    impact_coeff: float = 0.01,
    horizon: float = 1.0,
) -> float:
    score = toxic_order_score(order_flow_model, [], order_time, order_mark, horizon=horizon)
    return impact_coeff * score


