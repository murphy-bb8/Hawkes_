import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional


@dataclass
class ExogenousDesign:
    """
    Piecewise-constant exogenous feature design X(t) over [0, T].

    We represent X(t) by breakpoints 0 = b[0] < b[1] < ... < b[M] = T and
    constant feature vectors X_m for t in [b[m], b[m+1]).

    This enables exact integrals needed by Cox×Hawkes likelihood and gradients:
      ∫_0^T exp(θ_i^T X(t)) dt = Σ_m exp(θ_i^T X_m) * (b[m+1]-b[m])
      ∫_0^T exp(θ_i^T X(t)) X(t) dt = Σ_m X_m * exp(θ_i^T X_m) * (b[m+1]-b[m])
    """

    breakpoints: np.ndarray  # shape (M+1,)
    features: np.ndarray     # shape (M, K)
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    @property
    def num_segments(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    def value_at(self, t: float) -> np.ndarray:
        # 容错：将边界外的 t 夹到区间 [b0, bM]
        b0 = float(self.breakpoints[0])
        bM = float(self.breakpoints[-1])
        if t <= b0:
            idx = 0
        elif t >= bM:
            # t 恰好等于末端（或轻微越界）时，使用最后一个 piece
            idx = self.num_segments - 1
        else:
            # idx is the last breakpoint <= t
            idx = int(np.searchsorted(self.breakpoints, t, side="right") - 1)
            if idx == self.num_segments:
                idx -= 1
        x = self.features[idx]
        if self.mean_ is not None and self.std_ is not None:
            return (x - self.mean_) / self.std_
        return x

    def integral_exp_theta(self, theta_i: np.ndarray) -> float:
        # Σ_m exp(θ_i^T X_m) * Δt_m (with clipping to avoid overflow)
        X = self.features
        if self.mean_ is not None and self.std_ is not None:
            X = (X - self.mean_) / self.std_
        logits = X @ theta_i
        logits = np.clip(logits, -50.0, 50.0)
        deltas = np.diff(self.breakpoints)
        return float(np.sum(np.exp(logits) * deltas))

    def integral_exp_theta_times_X(self, theta_i: np.ndarray) -> np.ndarray:
        # Σ_m X_m * exp(θ_i^T X_m) * Δt_m  -> shape (K,)
        X = self.features
        if self.mean_ is not None and self.std_ is not None:
            X = (X - self.mean_) / self.std_
        logits = X @ theta_i
        logits = np.clip(logits, -50.0, 50.0)
        deltas = np.diff(self.breakpoints)
        weights = np.exp(logits) * deltas
        if self.mean_ is not None and self.std_ is not None:
            X_scaled = (self.features - self.mean_) / self.std_
            return (X_scaled * weights[:, None]).sum(axis=0)
        return (self.features * weights[:, None]).sum(axis=0)

    def integral_exp_theta_between(self, theta_i: np.ndarray, a: float, b: float) -> float:
        """Exact ∫_a^b exp(θ_i^T X(t)) dt under piecewise-constant X(t).
        Handles arbitrary [a,b] spanning multiple segments and clamps to design range.
        """
        b0 = float(self.breakpoints[0])
        bM = float(self.breakpoints[-1])
        if b <= a:
            return 0.0
        a = max(a, b0)
        b = min(b, bM)
        if b <= a:
            return 0.0
        # helper to scale a single feature vector
        def _scale_vec(x: np.ndarray) -> np.ndarray:
            if self.mean_ is not None and self.std_ is not None:
                return (x - self.mean_) / self.std_
            return x
        left = int(np.searchsorted(self.breakpoints, a, side="right") - 1)
        right = int(np.searchsorted(self.breakpoints, b, side="right") - 1)
        # Clamp indices to valid segment range [0, num_segments-1]
        n_seg = self.num_segments
        if n_seg <= 0:
            return 0.0
        left = max(0, min(left, n_seg - 1))
        right = max(0, min(right, n_seg - 1))
        total = 0.0
        for m in range(left, right + 1):
            seg_start = max(a, float(self.breakpoints[m]))
            # m+1 is safe since m <= n_seg-1 and breakpoints has length n_seg+1
            seg_end = min(b, float(self.breakpoints[m + 1]))
            dt = seg_end - seg_start
            if dt <= 0:
                continue
            x = _scale_vec(self.features[m])
            logit = float(np.clip(x @ theta_i, -50.0, 50.0))
            total += math.exp(logit) * dt
        return float(total)


def _rolling_count_within(times: np.ndarray, window: float, t_grid: np.ndarray) -> np.ndarray:
    counts = np.zeros_like(t_grid)
    j_start = 0
    for i, t in enumerate(t_grid):
        while j_start < len(times) and times[j_start] < t - window:
            j_start += 1
        j_end = j_start
        while j_end < len(times) and times[j_end] <= t:
            j_end += 1
        counts[i] = j_end - j_start
    return counts


def build_proxy_exogenous(events: List[Tuple[float, int]], T: float, dim: int, window: float = 1.0,
                          include: Sequence[str] = ("flow+", "flow-", "rv"), standardize: bool = True,
                          eps: float = 1e-6) -> ExogenousDesign:
    """
    Build piecewise-constant exogenous features from events as proxies when LOB/tick data
    is unavailable. Features are computed on a regular grid at step = window and held
    constant piecewise.

    Features per time t_k:
      - flow+_d: count of events of dimension d in (t_k - window, t_k]
      - flow-_d: count of events of all other dimensions in (t_k - window, t_k]
      - rv: realized volatility proxy = total event count in the window (rate proxy)

    Returns ExogenousDesign with K = (#dims if flow+ in include) + (#dims if flow- in include)
    + (1 if rv in include).
    """
    events_sorted = sorted(events, key=lambda x: x[0])
    t_grid = np.arange(0.0, T + 1e-12, window, dtype=float)
    if t_grid[-1] < T:
        t_grid = np.append(t_grid, T)

    by_dim: List[List[float]] = [[] for _ in range(dim)]
    for t, i in events_sorted:
        if 0.0 <= t <= T:
            by_dim[i].append(t)
    by_dim = [np.asarray(ts, dtype=float) for ts in by_dim]

    feature_cols: List[np.ndarray] = []
    names: List[str] = []

    if "flow+" in include:
        for d in range(dim):
            col = _rolling_count_within(by_dim[d], window, t_grid)
            feature_cols.append(col)
            names.append(f"flow_plus_{d}")
    if "flow-" in include:
        for d in range(dim):
            others = np.concatenate([by_dim[j] for j in range(dim) if j != d]) if dim > 1 else np.array([], dtype=float)
            col = _rolling_count_within(others, window, t_grid)
            feature_cols.append(col)
            names.append(f"flow_minus_{d}")
    if "rv" in include:
        all_times = np.sort(np.concatenate(by_dim)) if dim > 0 else np.array([], dtype=float)
        col = _rolling_count_within(all_times, window, t_grid)
        feature_cols.append(col)
        names.append("rv")

    if not feature_cols:
        # At least include a constant if nothing selected
        feature_cols.append(np.ones_like(t_grid))
        names.append("const")

    X_grid = np.vstack(feature_cols).T  # shape (M+1, K)
    # Use piecewise-constant values on [t_k, t_{k+1})
    breakpoints = t_grid.astype(float)
    features = X_grid[:-1, :].astype(float)
    exo = ExogenousDesign(breakpoints=breakpoints, features=features)
    if standardize and features.size > 0:
        mu = features.mean(axis=0)
        sd = features.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        exo.mean_ = mu
        exo.std_ = sd
    return exo


def with_intercept(exo: ExogenousDesign, value: float = 1.0) -> ExogenousDesign:
    """Return a new ExogenousDesign with a constant intercept column appended.
    The intercept is not standardized (mean=0, std=1), so θ_intercept acts as a pure intercept.
    """
    feats = exo.features
    ones = np.full((feats.shape[0], 1), float(value))
    feats2 = np.hstack([ones, feats])
    bp = exo.breakpoints.copy()
    out = ExogenousDesign(breakpoints=bp, features=feats2)
    if exo.mean_ is not None and exo.std_ is not None:
        mean2 = np.concatenate([np.array([0.0]), exo.mean_])
        std2 = np.concatenate([np.array([1.0]), exo.std_])
        out.mean_ = mean2
        out.std_ = std2
    else:
        out.mean_ = np.array([0.0] + [0.0] * feats.shape[1], dtype=float)
        out.std_ = np.array([1.0] + [1.0] * feats.shape[1], dtype=float)
    return out


# ---- Time-based exogenous builders ----
def build_time_exogenous(T: float, step: float = 1.0, period: float = 60.0,
                         components: Sequence[str] = ("sin", "cos")) -> ExogenousDesign:
    """
    Build piecewise-constant time exogenous features on a regular grid.
    - components can include 'sin' and/or 'cos'.
    - No intercept is included; use with_intercept afterwards.
    """
    t_grid = np.arange(0.0, T + 1e-12, step, dtype=float)
    if t_grid[-1] < T:
        t_grid = np.append(t_grid, T)
    centers = 0.5 * (t_grid[:-1] + t_grid[1:])
    cols = []
    for c in components:
        if c == 'sin':
            cols.append(np.sin(2.0 * np.pi * centers / max(period, 1e-8)))
        elif c == 'cos':
            cols.append(np.cos(2.0 * np.pi * centers / max(period, 1e-8)))
        else:
            raise ValueError(f"Unknown time component: {c}")
    if not cols:
        cols.append(np.ones_like(centers))
    X = np.vstack(cols).T.astype(float)  # (M, K)
    return ExogenousDesign(breakpoints=t_grid.astype(float), features=X)

