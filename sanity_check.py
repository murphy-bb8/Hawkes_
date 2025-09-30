from workflow.io import load_events_json
from workflow.tick_integration import fit_with_tick
from workflow.fit import fit_hawkes_exponential
from workflow.models.hawkes import HawkesExponential


def main():
    try:
        events = load_events_json("events.json")
    except FileNotFoundError:
        print("未找到 events.json，请先运行 simulate 或将文件放在项目根目录。")
        return
    print(f"事件数: {len(events)}")

    res = fit_with_tick(events, dim=1, T=30.0, lasso=0.05)
    if res is None:
        print("tick 未安装或运行失败，回退到内置MLE对照。可选安装: pip install tick")
        fit = fit_hawkes_exponential(events, T=30.0, dim=1, max_iter=1000)
        print({
            'mu': fit.mu.tolist(),
            'alpha': fit.alpha.tolist(),
            'beta': fit.beta.tolist(),
            'loglik': float(fit.loglik),
            'converged': bool(fit.converged),
            'iters': int(fit.n_iter),
        })
    else:
        print(res)


if __name__ == "__main__":
    main()