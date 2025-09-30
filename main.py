import argparse
import numpy as np

from workflow.models.hawkes import HawkesExponential
from workflow.fit import fit_hawkes_exponential
from workflow.eval import compare_hawkes_poisson
from workflow.viz import plot_event_raster, plot_intensity, plot_residuals
from workflow.io import save_events_json, load_events_json
from workflow.tuning.grid import grid_search


def run_simulate(args: argparse.Namespace):
    dim = args.dim
    T = args.T
    seed = args.seed
    mu = np.full(dim, args.mu)
    alpha = np.full((dim, dim), args.alpha)
    beta = np.full((dim, dim), args.beta)
    model = HawkesExponential(mu, alpha, beta)
    events = model.simulate_ogata(T=T, seed=seed)
    # 可选：确保至少产生一定数量的事件
    if args.min_events is not None and args.min_events > 0 and len(events) < args.min_events:
        retries = 0
        while len(events) < args.min_events and retries < args.max_retries:
            retries += 1
            events = model.simulate_ogata(T=T, seed=seed + retries if seed is not None else None)
        if len(events) < args.min_events:
            print(f"提示：未能达到期望事件数（min_events={args.min_events}），当前仅 {len(events)} 个。")
    print(f"生成事件数: {len(events)}")
    # 先保存，再绘图（避免交互式窗口阻塞导致未落盘）
    if args.out:
        save_events_json(args.out, events)
        print(f"事件已保存到 {args.out}")
    if args.plot:
        grid = np.linspace(0, T, 500)
        intens = model.intensity_over_grid(events, grid)
        if getattr(args, 'no_show', False):
            plot_event_raster(events, dim=dim, T=T, title="Hawkes 仿真事件", savepath="sim_raster.png")
            plot_intensity(grid, intens, title="强度轨迹", savepath="sim_intensity.png")
            print("图像已保存为 sim_raster.png, sim_intensity.png")
        else:
            plot_event_raster(events, dim=dim, T=T, title="Hawkes 仿真事件")
            plot_intensity(grid, intens, title="强度轨迹")
    return events, model


def run_fit(args: argparse.Namespace):
    # load or simulate
    if args.input:
        events = load_events_json(args.input)
        print(f"载入事件数: {len(events)}")
        # create a dummy model for plotting intensity baseline
        true_model = HawkesExponential(np.full(args.dim, args.mu), np.full((args.dim, args.dim), args.alpha), np.full((args.dim, args.dim), args.beta))
    else:
        events, true_model = run_simulate(args)
    fit = fit_hawkes_exponential(
        events,
        T=args.T,
        dim=args.dim,
        max_iter=args.max_iter,
        step_mu=args.step_mu,
        step_alpha=args.step_alpha,
        step_beta=args.step_beta,
        min_beta=args.min_beta,
        l2_alpha=args.l2_alpha,
        rho_max=args.rho_max,
    )
    est_model = HawkesExponential(fit.mu, fit.alpha, fit.beta)
    print("拟合结果:")
    print("mu=", fit.mu)
    print("alpha=\n", fit.alpha)
    print("beta=\n", fit.beta)
    print("loglik=", fit.loglik, "converged=", fit.converged, "iters=", fit.n_iter)

    # compare with Poisson baseline
    comp = compare_hawkes_poisson(events, args.T, fit.mu, fit.alpha, fit.beta)
    print("模型比较(AIC):", comp)

    # 先保存，再绘图（避免阻塞）
    if args.out:
        save_events_json(args.out, events)
        print(f"事件已保存到 {args.out}")

    if args.plot:
        grid = np.linspace(0, args.T, 500)
        intens = est_model.intensity_over_grid(events, grid)
        if getattr(args, 'no_show', False):
            plot_event_raster(events, dim=args.dim, T=args.T, title="事件与拟合模型", savepath="fit_raster.png")
            plot_intensity(grid, intens, title="拟合强度轨迹", savepath="fit_intensity.png")
            resids = est_model.compensate_residuals(events, args.T)
            plot_residuals(resids, savepath="fit_residuals.png")
            print("图像已保存为 fit_raster.png, fit_intensity.png, fit_residuals.png")
        else:
            plot_event_raster(events, dim=args.dim, T=args.T, title="事件与拟合模型")
            plot_intensity(grid, intens, title="拟合强度轨迹")
            resids = est_model.compensate_residuals(events, args.T)
            plot_residuals(resids)


def main():
    parser = argparse.ArgumentParser(description="Hawkes 过程建模与评估")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="仿真 Hawkes 过程")
    p_sim.add_argument("--dim", type=int, default=1)
    p_sim.add_argument("--T", type=float, default=10.0)
    p_sim.add_argument("--mu", type=float, default=0.2)
    p_sim.add_argument("--alpha", type=float, default=0.5)
    p_sim.add_argument("--beta", type=float, default=1.5)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--plot", action="store_true")
    p_sim.add_argument("--no_show", action="store_true", help="仅保存图片，不显示交互窗口")
    p_sim.add_argument("--min_events", type=int, default=1, help="仿真最少事件数（不足则重试）")
    p_sim.add_argument("--max_retries", type=int, default=10, help="重试次数上限")
    p_sim.add_argument("--out", type=str, default=None, help="保存仿真事件到JSON路径")
    p_sim.set_defaults(func=run_simulate)

    p_fit = sub.add_parser("fit", help="仿真并拟合 Hawkes 参数，比较泊松基线")
    p_fit.add_argument("--dim", type=int, default=1)
    p_fit.add_argument("--T", type=float, default=10.0)
    p_fit.add_argument("--mu", type=float, default=0.2)
    p_fit.add_argument("--alpha", type=float, default=0.5)
    p_fit.add_argument("--beta", type=float, default=1.5)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--plot", action="store_true")
    p_fit.add_argument("--no_show", action="store_true", help="仅保存图片，不显示交互窗口")
    p_fit.add_argument("--input", type=str, default=None, help="从JSON载入事件")
    p_fit.add_argument("--out", type=str, default=None, help="保存事件到JSON")
    # 稳定性相关可调参数
    p_fit.add_argument("--max_iter", type=int, default=600)
    p_fit.add_argument("--step_mu", type=float, default=1e-2)
    p_fit.add_argument("--step_alpha", type=float, default=1e-2)
    p_fit.add_argument("--step_beta", type=float, default=5e-4)
    p_fit.add_argument("--min_beta", type=float, default=0.1)
    p_fit.add_argument("--l2_alpha", type=float, default=0.0)
    p_fit.add_argument("--rho_max", type=float, default=0.95, help="分枝比上限（谱半径阈值）")
    p_fit.set_defaults(func=run_fit)

    # tune 子命令：在已有或仿真数据上做网格搜索
    p_tune = sub.add_parser("tune", help="在事件数据上做网格搜索以稳定化与选参")
    p_tune.add_argument("--dim", type=int, default=1)
    p_tune.add_argument("--T", type=float, default=10.0)
    p_tune.add_argument("--input", type=str, default=None, help="从JSON载入事件；若为空将先仿真")
    p_tune.add_argument("--mu", type=float, default=0.5)
    p_tune.add_argument("--alpha", type=float, default=0.6)
    p_tune.add_argument("--beta", type=float, default=1.2)
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.add_argument("--min_events", type=int, default=50)
    p_tune.add_argument("--max_retries", type=int, default=50)
    p_tune.add_argument("--out", type=str, default="events.json")
    p_tune.add_argument("--grid_min_beta", type=float, nargs='+', default=[0.4, 0.5, 0.6])
    p_tune.add_argument("--grid_l2_alpha", type=float, nargs='+', default=[0.0, 0.01, 0.02, 0.05])
    p_tune.add_argument("--grid_rho_max", type=float, nargs='+', default=[0.85, 0.9, 0.95])
    p_tune.add_argument("--max_iter", type=int, default=1500)
    p_tune.add_argument("--step_mu", type=float, default=1e-2)
    p_tune.add_argument("--step_alpha", type=float, default=1e-2)
    p_tune.add_argument("--step_beta", type=float, default=2e-4)

    def run_tune(args: argparse.Namespace):
        if args.input:
            events = load_events_json(args.input)
            T = args.T
        else:
            # 仿真生成更充足的数据
            sim_args = argparse.Namespace(
                dim=args.dim, T=args.T, seed=args.seed, mu=args.mu, alpha=args.alpha, beta=args.beta,
                plot=False, no_show=False, min_events=args.min_events, max_retries=args.max_retries, out=args.out
            )
            events, _ = run_simulate(sim_args)
            T = args.T
        report = grid_search(
            events, T, args.dim,
            grid_min_beta=args.grid_min_beta,
            grid_l2_alpha=args.grid_l2_alpha,
            grid_rho_max=args.grid_rho_max,
            max_iter=args.max_iter,
            step_mu=args.step_mu,
            step_alpha=args.step_alpha,
            step_beta=args.step_beta,
        )
        print("最优: ", report['best'])
        return report

    p_tune.set_defaults(func=run_tune)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
