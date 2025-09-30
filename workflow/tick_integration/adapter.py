from typing import List, Tuple, Optional
import numpy as np


def fit_with_tick(events: List[Tuple[float, int]], dim: int, T: float,
                  lasso: float = 0.0, max_iter: int = 1000):
    try:
        from tick.hawkes import HawkesExpKern, HawkesEM
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

        decays = 1.0
        try:
            # 先尝试最小二乘型的 HawkesExpKern
            learner = HawkesExpKern(
                decays=decays,
                penalty='l1' if lasso > 0 else None,
                C=lasso if lasso > 0 else 0.0,
                max_iter=max_iter,
                verbose=False,
            )
            learner.fit(times_by_dim)
            mu = np.array(learner.baseline, dtype=float)
            adj = np.array(learner.adjacency, dtype=float)
        except Exception as e1:
            # 回退到 EM，更鲁棒
            em = HawkesEM(decays=decays, max_iter=max_iter, tol=1e-6, verbose=False)
            em.fit(times_by_dim)
            mu = np.array(em.baseline, dtype=float)
            adj = np.array(em.adjacency, dtype=float)

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


