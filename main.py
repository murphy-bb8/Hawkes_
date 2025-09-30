import argparse
import numpy as np

from workflow.models.hawkes import HawkesExponential
from workflow.fit import fit_hawkes_exponential, map_em_exponential
from workflow.eval import compare_hawkes_poisson
from workflow.viz import plot_event_raster, plot_intensity, plot_residuals, plot_adjacency_heatmap
from workflow.io import save_events_json, load_events_json
from workflow.tuning.grid import grid_search
from workflow.gof import ks_exp_test, ks_uniform_test, ljung_box_test, compute_uniform_from_residuals
from workflow.gof import plot_gof_hist, plot_gof_qq
from workflow.preprocess import add_jitter_to_events, estimate_piecewise_mu


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
    if args.method == "map_em":
        res = map_em_exponential(
            events,
            T=args.T,
            dim=args.dim,
            init_mu=None,
            init_alpha=None,
            init_beta=None,
            max_iter=args.max_iter,
            min_beta=args.min_beta,
            prior_mu_a=args.prior_mu_a,
            prior_mu_b=args.prior_mu_b,
            prior_alpha_a=args.prior_alpha_a,
            prior_alpha_b=args.prior_alpha_b,
            prior_beta_a=args.prior_beta_a,
            prior_beta_b=args.prior_beta_b,
            update_beta=args.update_beta,
        )
        mu, alpha, beta = res.mu, res.alpha, res.beta
        est_model = HawkesExponential(mu, alpha, beta)
        print("MAP-EM 拟合结果:")
        print("mu=", mu)
        print("alpha=\n", alpha)
        print("beta=\n", beta)
        print("loglik=", res.loglik, "iters=", res.n_iter)
        comp = compare_hawkes_poisson(events, args.T, mu, alpha, beta)
    else:
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
        print("MLE 拟合结果:")
        print("mu=", fit.mu)
        print("alpha=\n", fit.alpha)
        print("beta=\n", fit.beta)
        print("loglik=", fit.loglik, "converged=", fit.converged, "iters=", fit.n_iter)
        comp = compare_hawkes_poisson(events, args.T, fit.mu, fit.alpha, fit.beta)

    # compare with Poisson baseline
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
            plot_adjacency_heatmap(est_model.alpha, threshold=getattr(args, 'adj_threshold', 0.0), savepath="fit_adjacency.png")
            print("图像已保存为 fit_raster.png, fit_intensity.png, fit_residuals.png, fit_adjacency.png")
        else:
            plot_event_raster(events, dim=args.dim, T=args.T, title="事件与拟合模型")
            plot_intensity(grid, intens, title="拟合强度轨迹")
            resids = est_model.compensate_residuals(events, args.T)
            plot_residuals(resids)
            plot_adjacency_heatmap(est_model.alpha, threshold=getattr(args, 'adj_threshold', 0.0))


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
    p_fit.add_argument("--method", type=str, default="mle", choices=["mle", "map_em"], help="拟合方法")
    p_fit.add_argument("--out", type=str, default=None, help="保存事件到JSON")
    # 稳定性相关可调参数
    p_fit.add_argument("--max_iter", type=int, default=600)
    p_fit.add_argument("--step_mu", type=float, default=1e-2)
    p_fit.add_argument("--step_alpha", type=float, default=1e-2)
    p_fit.add_argument("--step_beta", type=float, default=5e-4)
    p_fit.add_argument("--min_beta", type=float, default=0.1)
    p_fit.add_argument("--l2_alpha", type=float, default=0.0)
    p_fit.add_argument("--rho_max", type=float, default=0.95, help="分枝比上限（谱半径阈值）")
    p_fit.add_argument("--adj_threshold", type=float, default=0.0, help="稀疏传染图阈值（小于阈值置0）")
    # MAP-EM 先验与选项
    p_fit.add_argument("--prior_mu_a", type=float, default=1.0)
    p_fit.add_argument("--prior_mu_b", type=float, default=1.0)
    p_fit.add_argument("--prior_alpha_a", type=float, default=1.0)
    p_fit.add_argument("--prior_alpha_b", type=float, default=1.0)
    p_fit.add_argument("--prior_beta_a", type=float, default=1.0)
    p_fit.add_argument("--prior_beta_b", type=float, default=1.0)
    p_fit.add_argument("--update_beta", action="store_true", help="MAP-EM 中是否更新 beta（默认不更新）")
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

    # gof 子命令：做三件套 GOF 检验并出图
    p_gof = sub.add_parser("gof", help="对拟合残差做KS(Exp/Uniform)+Ljung-Box，并绘制QQ/直方图")
    p_gof.add_argument("--dim", type=int, default=1)
    p_gof.add_argument("--T", type=float, default=30.0)
    p_gof.add_argument("--input", type=str, required=True, help="事件JSON")
    p_gof.add_argument("--method", type=str, default="mle", choices=["mle", "map_em"])
    p_gof.add_argument("--jitter", action="store_true", help="对时间戳加微抖动")
    p_gof.add_argument("--seasonal_bins", type=int, default=0, help=">0 则用分段常数基线估计 mu 初值")
    p_gof.add_argument("--max_iter", type=int, default=2000)
    p_gof.add_argument("--step_mu", type=float, default=5e-3)
    p_gof.add_argument("--step_alpha", type=float, default=5e-3)
    p_gof.add_argument("--step_beta", type=float, default=1e-4)
    p_gof.add_argument("--min_beta", type=float, default=0.4)
    p_gof.add_argument("--rho_max", type=float, default=0.85)
    p_gof.add_argument("--prior_mu_a", type=float, default=1.0)
    p_gof.add_argument("--prior_mu_b", type=float, default=1.0)
    p_gof.add_argument("--prior_alpha_a", type=float, default=1.0)
    p_gof.add_argument("--prior_alpha_b", type=float, default=1.0)
    p_gof.add_argument("--prior_beta_a", type=float, default=2.0)
    p_gof.add_argument("--prior_beta_b", type=float, default=2.0)

    def run_gof(args: argparse.Namespace):
        events = load_events_json(args.input)
        if args.jitter:
            events = add_jitter_to_events(events, eps=1e-6)
        if args.seasonal_bins and args.seasonal_bins > 0:
            mu0 = estimate_piecewise_mu(events, args.T, args.dim, bins=args.seasonal_bins)
        else:
            mu0 = None
        # 拟合
        if args.method == 'map_em':
            res = map_em_exponential(
                events, T=args.T, dim=args.dim, init_mu=mu0,
                max_iter=args.max_iter, min_beta=args.min_beta,
                prior_mu_a=args.prior_mu_a, prior_mu_b=args.prior_mu_b,
                prior_alpha_a=args.prior_alpha_a, prior_alpha_b=args.prior_alpha_b,
                prior_beta_a=args.prior_beta_a, prior_beta_b=args.prior_beta_b,
            )
            mu, alpha, beta = res.mu, res.alpha, res.beta
        else:
            fit = fit_hawkes_exponential(
                events, T=args.T, dim=args.dim, max_iter=args.max_iter,
                step_mu=args.step_mu, step_alpha=args.step_alpha, step_beta=args.step_beta,
                min_beta=args.min_beta, l2_alpha=0.0, rho_max=args.rho_max,
            )
            mu, alpha, beta = fit.mu, fit.alpha, fit.beta
        model = HawkesExponential(mu, alpha, beta)
        # 残差与GOF
        resids = model.compensate_residuals(events, args.T)
        ks_exp = ks_exp_test(resids)
        u = compute_uniform_from_residuals(resids)
        ks_uni = ks_uniform_test(u)
        lb = ljung_box_test(u, lags=20)
        print({
            'n': len(resids),
            'KS_Exp': ks_exp,
            'KS_Uni': ks_uni,
            'LB': lb,
        })
        # 绘图
        plot_gof_hist(resids, savepath='gof_hist.png')
        plot_gof_qq(resids, savepath='gof_qq.png')

    p_gof.set_defaults(func=run_gof)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
