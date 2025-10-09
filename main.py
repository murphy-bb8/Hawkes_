import argparse
import numpy as np

from workflow.models.hawkes import HawkesExponential
from workflow.models.cox_hawkes import CoxHawkesExponential
from workflow.features.exogenous import build_proxy_exogenous
from workflow.fit.mle_exo import fit_cox_hawkes_theta
from workflow.fit.joint_exo import fit_cox_hawkes_joint
from workflow.fit.em_exo import em_update_alpha
from workflow.fit.map_em_exo import map_em_exogenous
from workflow.features.exogenous import with_intercept
from workflow.viz.plots_exo import plot_exogenous_trajectories
from workflow.data.bund import fetch_bund_events, load_bund_events_local
from workflow.fit import fit_hawkes_exponential, map_em_exponential
from workflow.eval import compare_hawkes_poisson
from workflow.viz import plot_event_raster, plot_intensity, plot_residuals, plot_adjacency_heatmap
from workflow.io import save_events_json, load_events_json
from workflow.tuning.grid import grid_search
from workflow.preprocess import add_jitter_to_events, estimate_piecewise_mu


def run_simulate(args: argparse.Namespace):
    dim = args.dim
    T = args.T
    seed = args.seed
    mu = np.full(dim, args.mu)
    alpha = np.full((dim, dim), args.alpha)
    beta = np.full((dim, dim), args.beta)
    if getattr(args, 'model', 'hawkes') == 'cox_hawkes':
        # build a trivial exogenous with proxies from simulated events later; first simulate using constant baseline
        model = HawkesExponential(mu, alpha, beta)
    else:
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
    if args.input == 'bund':
        # Prefer local path if provided via --bund_path
        if getattr(args, 'bund_path', None):
            events, T_bund = load_bund_events_local(args.bund_path, day=None)
        else:
            events, T_bund = fetch_bund_events(day=None)
        print(f"载入 Bund 事件数: {len(events)}，T≈{T_bund:.2f}s")
        # 若用户未指定 T，用 Bund 的时间范围
        if args.T is None or args.T <= 0:
            args.T = T_bund
        # auto dim=4 if not set
        if args.dim is None or args.dim <= 0:
            args.dim = 4
        true_model = HawkesExponential(np.full(args.dim, args.mu), np.full((args.dim, args.dim), args.alpha), np.full((args.dim, args.dim), args.beta))
    elif args.input:
        events = load_events_json(args.input)
        print(f"载入事件数: {len(events)}")
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

    # Optional: Cox×Hawkes baseline fitting for exogenous features
    if getattr(args, 'use_exo', False):
        exo = build_proxy_exogenous(events, args.T, args.dim, window=args.exo_window, standardize=args.exo_standardize)
        theta0 = np.zeros((args.dim, exo.num_features))
        if getattr(args, 'method', 'mle') == 'map_em_exo':
            exo_i = with_intercept(exo)
            res = map_em_exogenous(
                events, args.T, args.dim, exo_i,
                max_iter=args.exo_max_iter, step_theta=args.exo_step,
                min_beta=args.min_beta, rho_max=args.rho_max,
                prior_alpha_a=args.prior_alpha_a, prior_alpha_b=args.prior_alpha_b,
            )
            est_model = HawkesExponential(est_model.mu, res.alpha, res.beta)
            print("MAP-EM-Exo 结果: loglik=", res.loglik, " iters=", res.n_iter, " theta shape:", res.theta.shape)
        elif getattr(args, 'exo_joint', False):
            joint = fit_cox_hawkes_joint(
                events, args.T, args.dim, exo,
                init_theta=theta0, init_alpha=est_model.alpha, init_beta=est_model.beta,
                max_iter=args.exo_max_iter, step_theta=args.exo_step, step_alpha=args.step_alpha,
                step_beta=args.step_beta, min_beta=args.min_beta, l2_alpha=args.l2_alpha, rho_max=args.rho_max,
            )
            est_model = HawkesExponential(joint.theta @ 0 + est_model.mu, joint.alpha, joint.beta)  # keep Hawkes for plotting
            print("Exogenous joint (θ,α,β) 拟合: loglik=", joint.loglik, " iters=", joint.n_iter)
        else:
            exo_fit = fit_cox_hawkes_theta(events, args.T, args.dim, exo, est_model.alpha, est_model.beta, init_theta=theta0,
                                           step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                                           grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo)
            print("Exogenous baseline (Cox×Hawkes) θ 拟合:")
            print("theta shape:", exo_fit.theta.shape, " loglik:", exo_fit.loglik)
            if getattr(args, 'exo_em', False):
                emres = em_update_alpha(events, args.T, args.dim, exo_fit.theta, est_model.alpha, est_model.beta, max_iter=max(20, args.exo_max_iter // 10))
                est_model = HawkesExponential(est_model.mu, emres.alpha, emres.beta)
                print("EM alpha 更新完成: iters=", emres.n_iter)
        if args.plot:
            if getattr(args, 'no_show', False):
                plot_exogenous_trajectories(exo.breakpoints, exo.features, savepath='docs/img/exo_features.png')
            else:
                plot_exogenous_trajectories(exo.breakpoints, exo.features)

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

    p_fit = sub.add_parser("fit", help="仿真并拟合 Hawkes 参数，比较泊松基线；支持 Cox×Hawkes 外生项")
    p_fit.add_argument("--dim", type=int, default=1)
    p_fit.add_argument("--T", type=float, default=10.0)
    p_fit.add_argument("--mu", type=float, default=0.2)
    p_fit.add_argument("--alpha", type=float, default=0.5)
    p_fit.add_argument("--beta", type=float, default=1.5)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--plot", action="store_true")
    p_fit.add_argument("--no_show", action="store_true", help="仅保存图片，不显示交互窗口")
    p_fit.add_argument("--input", type=str, default=None, help="从JSON载入事件；或输入 'bund' 以加载 Bund 示例")
    p_fit.add_argument("--bund_path", type=str, default=None, help="本地 tick-datasets 根目录或 bund.npz 全路径")
    p_fit.add_argument("--model", type=str, default="hawkes", choices=["hawkes", "cox_hawkes"], help="模型类型")
    p_fit.add_argument("--use_exo", action="store_true", help="启用外生因子（Cox×Hawkes 基线）")
    p_fit.add_argument("--exo_window", type=float, default=1.0, help="外生 proxy 的时间窗口（秒）")
    p_fit.add_argument("--exo_step", type=float, default=1e-3, help="exo θ 的学习率")
    p_fit.add_argument("--exo_max_iter", type=int, default=300, help="exo 迭代次数（θ 或 θ,α,β）")
    p_fit.add_argument("--exo_joint", action="store_true", help="联合优化 θ 与 α/β")
    p_fit.add_argument("--exo_standardize", action="store_true", help="对外生特征做标准化")
    p_fit.add_argument("--grad_clip", type=float, default=10.0, help="梯度裁剪阈值")
    p_fit.add_argument("--lr_decay_exo", type=float, default=0.0, help="exo 学习率衰减系数")
    p_fit.add_argument("--exo_em", action="store_true", help="θ 拟合后，EM 更新 α")
    p_fit.add_argument("--method", type=str, default="mle", choices=["mle", "map_em", "map_em_exo"], help="拟合方法")
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
    p_gof.add_argument("--l2_alpha", type=float, default=0.0, help="对 alpha 的L2正则（稳定化）")
    # 可选：外生因子基线（Cox×Hawkes）以增强GOF
    p_gof.add_argument("--use_exo", action="store_true", help="使用基于代理特征的 Cox×Hawkes 基线进行 GOF")
    p_gof.add_argument("--exo_window", type=float, default=1.0, help="外生 proxy 的时间窗口（秒）")
    p_gof.add_argument("--exo_step", type=float, default=1e-3, help="exo θ 的学习率")
    p_gof.add_argument("--exo_max_iter", type=int, default=300, help="exo 迭代次数（θ）")
    p_gof.add_argument("--exo_standardize", action="store_true", help="对外生特征做标准化")
    p_gof.add_argument("--grad_clip", type=float, default=10.0, help="exo 梯度裁剪阈值")
    p_gof.add_argument("--lr_decay_exo", type=float, default=0.0, help="exo 学习率衰减系数")

    def run_gof(args: argparse.Namespace):
        # Lazy import to avoid requiring statsmodels unless GOF is used
        from workflow.gof import ks_exp_test, ks_uniform_test, ljung_box_test, compute_uniform_from_residuals
        from workflow.gof import plot_gof_hist, plot_gof_qq
        events = load_events_json(args.input)
        if args.jitter:
            events = add_jitter_to_events(events, eps=1e-6)
        if args.seasonal_bins and args.seasonal_bins > 0:
            mu0 = estimate_piecewise_mu(events, args.T, args.dim, bins=args.seasonal_bins)
        else:
            mu0 = None
        # 拟合（先得到 alpha/beta 与常数基线 mu 或用于 exo 的初值）
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
                min_beta=args.min_beta, l2_alpha=args.l2_alpha, rho_max=args.rho_max,
            )
            mu, alpha, beta = fit.mu, fit.alpha, fit.beta
        # 若使用外生因子基线，则在固定 α/β 的前提下拟合 θ，并用 Cox×Hawkes 计算残差
        if getattr(args, 'use_exo', False):
            from workflow.features.exogenous import build_proxy_exogenous
            from workflow.fit.mle_exo import fit_cox_hawkes_theta
            from workflow.models.cox_hawkes import CoxHawkesExponential
            exo = build_proxy_exogenous(
                events, args.T, args.dim,
                window=args.exo_window, standardize=args.exo_standardize,
            )
            theta0 = np.zeros((args.dim, exo.num_features))
            exo_fit = fit_cox_hawkes_theta(
                events, args.T, args.dim, exo,
                alpha=alpha, beta=beta, init_theta=theta0,
                step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo,
            )
            model = CoxHawkesExponential(exo_fit.theta, alpha, beta, exo)
        else:
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
