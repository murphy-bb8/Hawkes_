import json
import os
import sys
from typing import List, Tuple
import numpy as np


def load_events_from_json(filepath: str) -> List[Tuple[float, int]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(float(ev['t']), int(ev['i'])) for ev in data]


def build_times_by_dim(events: List[Tuple[float, int]], dim: int, T: float) -> List[np.ndarray]:
    times_by_dim_list: List[List[float]] = [[] for _ in range(dim)]
    for t, i in events:
        if 0.0 <= t <= T and 0 <= i < dim:
            times_by_dim_list[i].append(float(t))
    # strictly increasing per node, as tick expects
    eps = 1e-9
    out: List[np.ndarray] = []
    for lst in times_by_dim_list:
        if not lst:
            out.append(np.asarray([], dtype=np.float64))
            continue
        arr = np.asarray(sorted(lst), dtype=np.float64)
        for k in range(1, arr.shape[0]):
            if arr[k] <= arr[k - 1]:
                arr[k] = min(arr[k - 1] + eps, T - eps)
        arr = arr[arr < T]
        out.append(arr)
    return out


def main():
    try:
        from tick.hawkes import HawkesExpKern
    except Exception as e:
        print("[ERROR] 无法导入 tick.hawkes.HawkesExpKern，请确认当前 Python 环境已安装 tick 并已激活。\n原因:", e)
        sys.exit(1)

    data_path = os.path.join(os.path.dirname(__file__), 'events_100k.json')
    if not os.path.exists(data_path):
        print("[ERROR] 未找到 events_100k.json：", data_path)
        sys.exit(1)

    print("加载数据...")
    events = load_events_from_json(data_path)
    T = float(events[-1][0]) if events else 0.0
    dim = (max(i for _, i in events) + 1) if events else 1
    print(f"事件数={len(events)}, 维度={dim}, T≈{T:.4f}")

    print("构造 tick 输入格式...")
    times_by_dim = build_times_by_dim(events, dim, T)
    for d in range(dim):
        print(f"  节点{d}: {times_by_dim[d].size} 个事件")

    # 使用对数似然目标，且 penalty='none' 时也需给正的 C
    beta = 1.5
    # 当不使用惩罚时，一些版本不会开启正约束，改用极弱的L2以启用正投影
    learner = HawkesExpKern(decays=beta, gofit='least-squares', penalty='l2', C=1e12, max_iter=500, verbose=True)

    print("开始拟合（HawkesExpKern, MLE）...")
    learner.fit(times_by_dim)

    mu = np.asarray(learner.baseline, dtype=float)
    alpha = np.asarray(learner.adjacency, dtype=float)
    print("拟合完成：")
    print("  mu.shape=", mu.shape)
    print("  alpha.shape=", alpha.shape)
    print("  mu前几项=", mu[:min(5, mu.size)])
    print("  alpha前几项=", alpha.ravel()[:min(10, alpha.size)])

    ll = float(learner.score(times_by_dim, end_times=T))
    k_params = int(dim + dim * dim)  # beta 固定
    aic = float(2 * k_params - 2 * ll)
    print(f"对数似然={ll:.6f}, AIC={aic:.6f}, k={k_params}")


if __name__ == '__main__':
    main()
