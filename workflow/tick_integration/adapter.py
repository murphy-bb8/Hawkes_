from typing import List, Tuple, Optional
import numpy as np


def fit_with_tick(
    events: List[Tuple[float, int]],
    dim: int,
    T: float,
    decays: float = 1.0,
    lasso: float = 0.0,
    max_iter: int = 1000,
    verbose: bool = False,
):
    try:
        from tick.hawkes import HawkesExpKern
    except Exception as e:
        print("未安装 tick，跳过 tick 拟合：", e)
        return None

    try:
        # tick 需要按维度提供 numpy.ndarray（dense, 严格递增）
        times_by_dim_list: List[List[float]] = [[] for _ in range(dim)]
        for t, i in sorted(events, key=lambda x: x[0]):
            if 0.0 <= t <= T:
                times_by_dim_list[i].append(float(t))

        eps = 1e-9
        times_by_dim: List[np.ndarray] = []
        for lst in times_by_dim_list:
            if not lst:
                times_by_dim.append(np.asarray([], dtype=np.float64))
                continue
            arr = np.asarray(sorted(lst), dtype=np.float64)
            # 使严格递增，避免重复时间戳
            for k in range(1, arr.shape[0]):
                if arr[k] <= arr[k - 1]:
                    arr[k] = min(arr[k - 1] + eps, T - eps)
            # 过滤超出 T 的
            arr = arr[arr < T]
            times_by_dim.append(arr)

        # 仅用 HawkesExpKern（参数化指数核 MLE），失败则直接返回 None 由外层回退
        # 注意：tick 的 C 必须为正数，即使 penalty='none'；并显式使用对数似然目标
        # Some versions of tick expect decays as positional, not keyword
        # 对于某些 tick 版本，penalty='none' 不应用正约束，会触发影响和为负的错误。
        # 且极小的 L1（C 很小）可能在内部引发除零；因此当 lasso 很小（≤1e-6）时转用温和的 L2。
        small_l1 = (lasso is not None and lasso <= 1e-6)
        use_penalty = 'l1' if (lasso and lasso > 1e-6) else 'l2'
        # C 是 1/strength；L2 用较大 C≈1e6 表示极弱正则以启用正投影
        use_C = (float(lasso) if (lasso and lasso > 1e-6) else 1e6)
        learner = HawkesExpKern(
            float(decays),
            gofit='least-squares',
            penalty=use_penalty,
            C=use_C,
            max_iter=int(max_iter),
            verbose=bool(verbose),
        )
        learner.fit(times_by_dim)
        # Robust extraction across versions
        if hasattr(learner, 'baseline'):
            mu = np.array(learner.baseline, dtype=float)
        elif hasattr(learner, 'baseline_'):
            mu = np.array(learner.baseline_, dtype=float)
        elif hasattr(learner, 'get_baseline'):
            mu = np.array(learner.get_baseline(), dtype=float)
        else:
            raise AttributeError('tick learner baseline attribute not found')
        if hasattr(learner, 'adjacency'):
            adj = np.array(learner.adjacency, dtype=float)
        elif hasattr(learner, 'adjacency_'):
            adj = np.array(learner.adjacency_, dtype=float)
        elif hasattr(learner, 'get_adjacency'):
            adj = np.array(learner.get_adjacency(), dtype=float)
        else:
            raise AttributeError('tick learner adjacency attribute not found')

        beta = np.full_like(adj, float(decays))
        return {
            'mu': mu,
            'alpha': adj,
            'beta': beta,
            'baseline': mu.tolist(),
            'adjacency': adj.tolist(),
            'decays': float(decays),
        }
    except Exception as e:
        print("tick 拟合失败，回退到内置MLE：", e)
        return None


