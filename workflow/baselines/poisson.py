import math
import numpy as np
from typing import List, Tuple, Optional


class PoissonProcess:
    def __init__(self, lambda_rate: float):
        if lambda_rate <= 0:
            raise ValueError("lambda_rate must be > 0")
        self.lambda_rate = float(lambda_rate)

    def simulate(self, T: float, seed: Optional[int] = None) -> List[float]:
        rng = np.random.default_rng(seed)
        t = 0.0
        times: List[float] = []
        while t < T:
            w = rng.exponential(1.0 / self.lambda_rate)
            t += w
            if t <= T:
                times.append(t)
        return times

    def loglik(self, times: List[float], T: float) -> float:
        n = len(times)
        return float(n * math.log(self.lambda_rate) - self.lambda_rate * T)


def mle_rate_from_times(times: List[float], T: float) -> float:
    return max(1e-12, len(times) / max(T, 1e-12))


def marked_events_to_times(events: List[Tuple[float, int]]) -> List[float]:
    return [float(t) for t, _ in events]


