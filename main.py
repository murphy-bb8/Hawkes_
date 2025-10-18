import argparse
import numpy as np
try:
    import tick.hawkes  # ensure tick modules are importable in spawned workers
except Exception:
    tick = None  # optional: allow environments without tick

from workflow.models.hawkes import HawkesExponential
from workflow.models.cox_hawkes import CoxHawkesExponential
from workflow.features.exogenous import build_proxy_exogenous
from workflow.fit.mle_exo import fit_cox_hawkes_theta
from workflow.fit.joint_exo import fit_cox_hawkes_joint
from workflow.fit.em_exo import em_update_alpha
from workflow.fit.map_em_exo import map_em_exogenous
from workflow.features.exogenous import with_intercept
from workflow.features.exogenous import build_time_exogenous
from workflow.viz.plots_exo import plot_exogenous_trajectories
from workflow.data.bund import fetch_bund_events, load_bund_events_local
from workflow.fit import fit_hawkes_exponential, map_em_exponential
from workflow.eval import compare_hawkes_poisson
from workflow.viz import plot_event_raster, plot_intensity, plot_residuals, plot_adjacency_heatmap
from workflow.io import save_events_json, load_events_json
from workflow.tuning.grid import grid_search
from workflow.preprocess import add_jitter_to_events, estimate_piecewise_mu
from workflow.tick_integration import fit_with_tick


def _exp_worker_seed(payload):
    """Worker that runs one seed with serial beta grid search for Hawkes and Full.
    Returns a results dict for this seed, including GOF plots saved with seed/model suffixes.
    """
    import numpy as np
    from workflow.models.hawkes import HawkesExponential
    from workflow.models.cox_hawkes import CoxHawkesExponential
    from workflow.features.exogenous import build_proxy_exogenous, with_intercept, build_time_exogenous
    from workflow.fit.mle_exo import fit_cox_hawkes_theta
    from workflow.fit.em_exo import em_update_alpha
    from workflow.tick_integration import fit_with_tick
    from workflow.fit.mle import fit_hawkes_exponential
    from workflow.gof import ks_exp_test, ks_uniform_test, ljung_box_test, compute_uniform_from_residuals
    from workflow.gof import plot_gof_hist, plot_gof_qq
    seed = payload['seed']
    dim = payload['dim']
    beta_grid = payload['beta_grid']
    tick_lasso = payload['tick_lasso']
    tick_max_iter = payload['tick_max_iter']
    exo_window = payload['exo_window']
    exo_step = payload['exo_step']
    exo_max_iter = payload['exo_max_iter']
    exo_standardize = payload['exo_standardize']
    exo_kind = payload.get('exo_kind', 'proxy')
    period = payload.get('period', 30.0)
    exo_step_sim = payload.get('exo_step_sim', 0.5)
    train_events = payload['train_events']
    val_events = payload['val_events']
    T_train = payload['T_train']
    T_val = payload['T_val']

    # helper: fit hawkes for a fixed beta using tick if available, otherwise fallback to internal MLE
    def _fit_hawkes_fixed_beta(b_val: float):
        tick_res = fit_with_tick(train_events, dim, T_train, decays=float(b_val), lasso=tick_lasso, max_iter=tick_max_iter, verbose=False)
        if tick_res is not None:
            return tick_res['mu'], tick_res['alpha'], tick_res['beta']
        # fallback: internal MLE with beta fixed
        beta_fixed = np.full((dim, dim), float(b_val), dtype=float)
        fit = fit_hawkes_exponential(
            train_events,
            T=T_train,
            dim=dim,
            init_beta=beta_fixed,
            max_iter=800,
            step_mu=1e-2,
            step_alpha=1e-2,
            step_beta=0.0,
            min_beta=float(b_val),
            l2_alpha=0.0,
            rho_max=0.95,
        )
        return fit.mu, fit.alpha, fit.beta

    # Hawkes: grid over beta (fixed)
    best_h = None
    for b in beta_grid:
        mu_b, alpha_b, beta_mat_b = _fit_hawkes_fixed_beta(b)
        model_h = HawkesExponential(mu_b, alpha_b, beta_mat_b)
        ll = model_h.loglikelihood(val_events, T_val)
        k_h = int(dim + dim * dim)
        aic = float(2 * k_h - 2 * ll)
        cand = {'beta': float(b), 'll': ll, 'aic': aic, 'mu': mu_b, 'alpha': alpha_b, 'beta_mat': beta_mat_b}
        if best_h is None or cand['aic'] < best_h['aic']:
            best_h = cand
    hawkes = HawkesExponential(best_h['mu'], best_h['alpha'], best_h['beta_mat'])
    ll_h = best_h['ll']
    aic_h = best_h['aic']

    # Full: grid over beta (fixed)
    best_f = None
    for b in beta_grid:
        mu_b, alpha_b, beta_mat_b = _fit_hawkes_fixed_beta(b)
        if exo_kind == 'time':
            exo = build_time_exogenous(T_train, step=exo_step_sim, period=period, components=("sin","cos"))
        else:
            exo = build_proxy_exogenous(train_events, T_train, dim, window=exo_window, standardize=exo_standardize)
        exo_i = with_intercept(exo)
        theta0 = np.zeros((dim, exo_i.num_features))
        theta0[:, 0] = np.log(np.maximum(mu_b, 1e-8))
        exo_fit = fit_cox_hawkes_theta(train_events, T_train, dim, exo_i, alpha=alpha_b, beta=beta_mat_b, init_theta=theta0,
                                       step=exo_step, max_iter=exo_max_iter, adam=True, grad_clip=10.0, lr_decay=0.0)
        emres = em_update_alpha(train_events, T_train, dim, exo_fit.theta, alpha_b, beta_mat_b, max_iter=max(20, exo_max_iter // 10))
        if exo_kind == 'time':
            exo_val = build_time_exogenous(T_val, step=exo_step_sim, period=period, components=("sin","cos"))
        else:
            exo_val = build_proxy_exogenous(val_events, T_val, dim, window=exo_window, standardize=exo_standardize)
        exo_val_i = with_intercept(exo_val)
        full_model = CoxHawkesExponential(exo_fit.theta, emres.alpha, emres.beta, exo_val_i)
        ll = full_model.loglikelihood(val_events, T_val)
        k_f = int(dim * exo_i.num_features + dim * dim)
        aic = float(2 * k_f - 2 * ll)
        cand_f = {
            'beta': float(b), 'll': ll, 'aic': aic,
            'theta': exo_fit.theta, 'alpha': emres.alpha, 'beta_mat': emres.beta,
            'K': exo_i.num_features,
        }
        if best_f is None or cand_f['aic'] < best_f['aic']:
            best_f = cand_f

    # GOF for selected Hawkes and Full
    resids_h = hawkes.compensate_residuals(val_events, T_val)
    ks_h = ks_exp_test(resids_h)
    u_h = compute_uniform_from_residuals(resids_h)
    lb_h = ljung_box_test(u_h, lags=20)
    if exo_kind == 'time':
        exo_val = build_time_exogenous(T_val, step=exo_step_sim, period=period, components=("sin","cos"))
    else:
        exo_val = build_proxy_exogenous(val_events, T_val, dim, window=exo_window, standardize=exo_standardize)
    exo_val_i = with_intercept(exo_val)
    full = CoxHawkesExponential(best_f['theta'], best_f['alpha'], best_f['beta_mat'], exo_val_i)
    resids_f = full.compensate_residuals(val_events, T_val)
    ks_f = ks_exp_test(resids_f)
    u_f = compute_uniform_from_residuals(resids_f)
    lb_f = ljung_box_test(u_f, lags=20)
    # Save plots
    plot_gof_hist(resids_h, savepath=f'gof_hist_seed{seed}_hawkes.png')
    plot_gof_qq(resids_h, savepath=f'gof_qq_seed{seed}_hawkes.png')
    plot_gof_hist(resids_f, savepath=f'gof_hist_seed{seed}_full.png')
    plot_gof_qq(resids_f, savepath=f'gof_qq_seed{seed}_full.png')

    return {
        'seed': seed,
        'hawkes': {'ll': ll_h, 'aic': aic_h, 'KS': ks_h, 'LB': lb_h, 'beta': best_h['beta']},
        'full': {'ll': best_f['ll'], 'aic': best_f['aic'], 'KS': ks_f, 'LB': lb_f, 'K': best_f['K'], 'beta': best_f['beta']},
    }


def _expected_T_for_min_events(dim: int, mu: float, alpha: float, beta: float, min_events: int) -> float:
    # Approximate E[N] ≈ (sum mu) * T / (1 - rho), with rho ≈ spectral radius of alpha/beta.
    if min_events is None or min_events <= 0:
        return 0.0
    sum_mu = float(mu) * max(dim, 1)
    G = (alpha / max(beta, 1e-8))
    rho = min(0.99, abs(G))  # univariate approx; dim>1 uses scalar alpha here
    denom = max(1e-8, 1.0 - rho)
    return float(min_events / max(sum_mu / denom, 1e-8))


def _split_and_shift_events(events, T_total: float, split_train: float, split_val: float):
    assert abs((split_train + split_val) - 1.0) < 1e-6, "split_train + split_val must be 1.0"
    t_cut = float(T_total * split_train)
    train = []
    val = []
    for t, i in sorted(events, key=lambda x: x[0]):
        if t < t_cut:
            train.append((float(t), int(i)))
        else:
            val.append((float(t - t_cut), int(i)))
    return train, t_cut, val, float(T_total - t_cut)


def run_simulate(args: argparse.Namespace):
    dim = args.dim
    T = args.T
    seed = args.seed
    mu = np.full(dim, args.mu)
    alpha = np.full((dim, dim), args.alpha)
    beta = np.full((dim, dim), args.beta)
    # 选择仿真模型：
    # - hawkes: 常数基线纯 Hawkes
    # - cox_hawkes: 时间变基线的 Cox×Hawkes（用时间特征或 proxy 构造）
    sim_model = getattr(args, 'model', 'hawkes')
    if sim_model == 'cox_hawkes':
        # 预构造 theta；exo 需与当前仿真 T 对齐，后续针对 T_cur 现场构造
        period = getattr(args, 'period', 30.0)
        step = getattr(args, 'exo_step_sim', 0.5)
        # 先以一个最小窗口构造以确定 K
        proto_exo_i = with_intercept(build_time_exogenous(max(T, 1.0), step=step, period=period, components=("sin", "cos")))
        K = proto_exo_i.num_features
        theta = np.zeros((dim, K), dtype=float)
        theta[:, 0] = np.log(np.maximum(mu, 1e-8))
        if K >= 3:
            theta[:, 1] = getattr(args, 'sin_amp', 1.0)
            theta[:, 2] = getattr(args, 'cos_amp', 0.5)
        # helper: build model for a given horizon
        def build_model_for(horizon: float):
            exo_i = with_intercept(build_time_exogenous(max(horizon, step), step=step, period=period, components=("sin", "cos")))
            return CoxHawkesExponential(theta, alpha, beta, exo_i), exo_i
    else:
        model = HawkesExponential(mu, alpha, beta)
    # 智能达到目标事件数：自适应放大 T
    events = []
    T_cur = T
    if args.min_events is not None and args.min_events > 0:
        if T_cur <= 0:
            T_cur = max(1.0, _expected_T_for_min_events(dim, args.mu, args.alpha, args.beta, args.min_events))
        retries = 0
        while retries <= args.max_retries:
            if sim_model == 'cox_hawkes':
                _model_cur, _exo_cur = build_model_for(T_cur)
                events = _model_cur.simulate_ogata(T=T_cur, seed=seed + retries if seed is not None else None)
            else:
                events = model.simulate_ogata(T=T_cur, seed=seed + retries if seed is not None else None)
            if len(events) >= args.min_events:
                break
            # 放大时间窗后重仿真（指数放大，快速收敛到目标规模）
            T_cur *= 2.0
            retries += 1
        if len(events) < args.min_events:
            print(f"Warning: min_events={args.min_events} not reached, got {len(events)} with T≈{T_cur:.2f}.")
        # 截断到 min_events 并用最后事件时间作为 T
        if args.min_events and len(events) > args.min_events:
            events = events[:args.min_events]
        T = (events[-1][0] if events else T_cur)
    else:
        if sim_model == 'cox_hawkes':
            _model_cur, _exo_cur = build_model_for(T)
            events = _model_cur.simulate_ogata(T=T, seed=seed)
        else:
            events = model.simulate_ogata(T=T, seed=seed)
        # 未设置 min_events：直接按给定 T 仿真
    print(f"Generated events: {len(events)}")
    # 先保存，再绘图（避免交互式窗口阻塞导致未落盘）
    if args.out:
        save_events_json(args.out, events)
        print(f"Events saved to {args.out}")
    if args.plot:
        grid = np.linspace(0, T, 500)
        if sim_model == 'cox_hawkes':
            _model_plot, _ = build_model_for(T)
            intens = _model_plot.intensity_over_grid(events, grid)
        else:
            intens = model.intensity_over_grid(events, grid)
        if getattr(args, 'no_show', False):
            plot_event_raster(events, dim=dim, T=T, title="Simulated Hawkes events", savepath="sim_raster.png")
            plot_intensity(grid, intens, title="Intensity", savepath="sim_intensity.png")
            print("Saved: sim_raster.png, sim_intensity.png")
        else:
            plot_event_raster(events, dim=dim, T=T, title="Simulated Hawkes events")
            plot_intensity(grid, intens, title="Intensity")
    if sim_model == 'cox_hawkes':
        _model_final, _ = build_model_for(T)
        return events, _model_final
    else:
        return events, model


def run_fit(args: argparse.Namespace):
    # load or simulate
    if args.input == 'bund':
        # Prefer local path if provided via --bund_path
        if getattr(args, 'bund_path', None):
            events, T_bund = load_bund_events_local(args.bund_path, day=None)
        else:
            events, T_bund = fetch_bund_events(day=None)
        print(f"Loaded Bund events: {len(events)}, T≈{T_bund:.2f}s")
        # 若用户未指定 T，用 Bund 的时间范围
        if args.T is None or args.T <= 0:
            args.T = T_bund
        # auto dim=4 if not set
        if args.dim is None or args.dim <= 0:
            args.dim = 4
        true_model = HawkesExponential(np.full(args.dim, args.mu), np.full((args.dim, args.dim), args.alpha), np.full((args.dim, args.dim), args.beta))
    elif args.input:
        events = load_events_json(args.input)
        print(f"Loaded events: {len(events)}")
        # 若用户未指定 T 或 T<=0，则用事件时间范围
        if args.T is None or args.T <= 0:
            args.T = float(events[-1][0]) if events else 0.0
        true_model = HawkesExponential(np.full(args.dim, args.mu), np.full((args.dim, args.dim), args.alpha), np.full((args.dim, args.dim), args.beta))
    else:
        events, true_model = run_simulate(args)
    # 训练/验证切分（若提供）
    use_split = getattr(args, 'split_train', None) is not None and getattr(args, 'split_val', None) is not None
    if use_split:
        if abs(args.split_train + args.split_val - 1.0) > 1e-6:
            raise ValueError("--split_train 与 --split_val 之和必须为 1.0")
        train_events, T_train, val_events, T_val = _split_and_shift_events(events, args.T, args.split_train, args.split_val)
    else:
        train_events, T_train = events, args.T
        val_events, T_val = events, args.T

    if args.method == "map_em":
        res = map_em_exponential(
            train_events,
            T=T_train,
            dim=args.dim,
            init_mu=None,
            init_alpha=None,
            init_beta=(np.full((args.dim, args.dim), args.beta) if getattr(args, 'freeze_beta', False) else None),
            max_iter=args.max_iter,
            min_beta=args.min_beta,
            prior_mu_a=args.prior_mu_a,
            prior_mu_b=args.prior_mu_b,
            prior_alpha_a=args.prior_alpha_a,
            prior_alpha_b=args.prior_alpha_b,
            prior_beta_a=args.prior_beta_a,
            prior_beta_b=args.prior_beta_b,
            update_beta=(not getattr(args, 'freeze_beta', False)) if hasattr(args, 'freeze_beta') else args.update_beta,
        )
        mu, alpha, beta = res.mu, res.alpha, res.beta
        est_model = HawkesExponential(mu, alpha, beta)
        print("MAP-EM fit:")
        print("mu=", mu)
        print("alpha=\n", alpha)
        print("beta=\n", beta)
        print("loglik=", res.loglik, "iters=", res.n_iter)
        comp = compare_hawkes_poisson(val_events, T_val, mu, alpha, beta)
    elif args.method == "mle":
        fit = fit_hawkes_exponential(
            train_events,
            T=T_train,
            dim=args.dim,
            init_beta=(np.full((args.dim, args.dim), args.beta) if getattr(args, 'freeze_beta', False) else None),
            max_iter=args.max_iter,
            step_mu=args.step_mu,
            step_alpha=args.step_alpha,
            step_beta=0.0 if getattr(args, 'freeze_beta', False) else args.step_beta,
            min_beta=args.min_beta,
            l2_alpha=args.l2_alpha,
            rho_max=args.rho_max,
        )
        est_model = HawkesExponential(fit.mu, fit.alpha, fit.beta)
        print("MLE fit:")
        print("mu=", fit.mu)
        print("alpha=\n", fit.alpha)
        print("beta=\n", fit.beta)
        print("loglik=", fit.loglik, "converged=", fit.converged, "iters=", fit.n_iter)
        comp = compare_hawkes_poisson(val_events, T_val, fit.mu, fit.alpha, fit.beta)
    elif args.method == "tick_mle":
        # Pure Hawkes via tick MLE with fixed beta (decays)
        tick_decays = float(getattr(args, 'tick_decays', args.beta))
        tick_res = fit_with_tick(
            train_events,
            dim=args.dim,
            T=T_train,
            decays=tick_decays,
            lasso=getattr(args, 'tick_lasso', 0.0),
            max_iter=getattr(args, 'tick_max_iter', 1000),
            verbose=False,
        )
        if tick_res is None:
            raise RuntimeError("tick 拟合失败或未安装 tick")
        mu = tick_res['mu']
        alpha = tick_res['alpha']
        beta = tick_res['beta']
        est_model = HawkesExponential(mu, alpha, beta)
        # Validation metrics
        ll = est_model.loglikelihood(val_events, T_val)
        k_params = int(args.dim + args.dim * args.dim)  # beta fixed
        aic = float(2 * k_params - 2 * ll)
        bic = float(k_params * np.log(max(len(val_events), 1)) - 2 * ll)
        print("tick MLE (Hawkes) validation:")
        print({
            'loglik': float(ll),
            'AIC': float(aic),
            'BIC': float(bic),
            'k': k_params,
        })
        comp = {'AIC': aic, 'loglik': ll}
    elif args.method == "tick_full_em":
        # Full model: use tick to estimate alpha with fixed beta, then fit exogenous theta (optionally theta-first), EM refine alpha
        tick_decays = float(getattr(args, 'tick_decays', args.beta))
        tick_res = fit_with_tick(
            train_events,
            dim=args.dim,
            T=T_train,
            decays=tick_decays,
            lasso=getattr(args, 'tick_lasso', 0.0),
            max_iter=getattr(args, 'tick_max_iter', 1000),
            verbose=False,
        )
        if tick_res is None:
            raise RuntimeError("tick 拟合失败或未安装 tick")
        alpha = tick_res['alpha']
        beta = tick_res['beta']  # fixed
        # Build exogenous design with intercept and fit theta
        if getattr(args, 'exo_kind', 'proxy') == 'time':
            exo = build_time_exogenous(T_train, step=getattr(args, 'exo_step_sim', 0.5), period=getattr(args, 'period', 30.0), components=("sin","cos"))
        else:
            exo = build_proxy_exogenous(train_events, T_train, args.dim, window=args.exo_window, standardize=args.exo_standardize)
        exo_i = with_intercept(exo)
        theta0 = np.zeros((args.dim, exo_i.num_features))
        if 'mu' in tick_res:
            theta0[:, 0] = np.log(np.maximum(tick_res['mu'], 1e-8))
        if getattr(args, 'theta_first', False):
            alpha_init = np.maximum(0.0, alpha) * float(getattr(args, 'alpha_init_scale', 0.1))
            exo_fit = fit_cox_hawkes_theta(
                train_events, T_train, args.dim, exo_i,
                alpha=alpha_init, beta=beta, init_theta=theta0,
                step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo,
            )
            emres = em_update_alpha(train_events, T_train, args.dim, exo_fit.theta, alpha_init, beta, max_iter=max(20, args.exo_max_iter // 10))
            alpha = emres.alpha
        else:
            exo_fit = fit_cox_hawkes_theta(
                train_events, T_train, args.dim, exo_i,
                alpha=alpha, beta=beta, init_theta=theta0,
                step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo,
            )
            if getattr(args, 'exo_em', True):
                emres = em_update_alpha(train_events, T_train, args.dim, exo_fit.theta, alpha, beta, max_iter=max(20, args.exo_max_iter // 10))
                alpha = emres.alpha
        # Validate on val window
        if getattr(args, 'exo_kind', 'proxy') == 'time':
            exo_val = build_time_exogenous(T_val, step=getattr(args, 'exo_step_sim', 0.5), period=getattr(args, 'period', 30.0), components=("sin","cos"))
        else:
            exo_val = build_proxy_exogenous(val_events, T_val, args.dim, window=args.exo_window, standardize=args.exo_standardize)
        exo_val_i = with_intercept(exo_val)
        model_full = CoxHawkesExponential(exo_fit.theta, alpha, beta, exo_val_i)
        ll = model_full.loglikelihood(val_events, T_val)
        k_params = int(args.dim * exo_i.num_features + args.dim * args.dim)  # beta fixed
        aic = float(2 * k_params - 2 * ll)
        bic = float(k_params * np.log(max(len(val_events), 1)) - 2 * ll)
        print("tick+EM (Full Cox×Hawkes) validation:")
        print({
            'loglik': float(ll),
            'AIC': float(aic),
            'BIC': float(bic),
            'k': k_params,
            'num_features_per_dim': exo_i.num_features,
        })
        est_model = HawkesExponential(np.maximum(1e-8, np.exp(exo_fit.theta[:, 0])), alpha, beta)  # for plotting only
        comp = {'AIC': aic, 'loglik': ll}
    else:
        raise ValueError(f"Unknown --method {args.method}")

    # Optional: Cox×Hawkes baseline fitting for exogenous features (skip if tick_full_em already did it)
    if getattr(args, 'use_exo', False) and args.method not in ("tick_full_em",):
        # 在各自窗口上构造外生特征
        exo = build_proxy_exogenous(train_events, T_train, args.dim, window=args.exo_window, standardize=args.exo_standardize)
        exo_i = with_intercept(exo)
        theta0 = np.zeros((args.dim, exo_i.num_features))
        # 截距用 log(mû) 初始化，保证嵌套纯 Hawkes
        theta0[:, 0] = np.log(np.maximum(est_model.mu, 1e-8))
        if getattr(args, 'method', 'mle') == 'map_em_exo':
            res = map_em_exogenous(
                train_events, T_train, args.dim, exo_i,
                max_iter=args.exo_max_iter, step_theta=args.exo_step,
                min_beta=args.min_beta, rho_max=args.rho_max,
                prior_alpha_a=args.prior_alpha_a, prior_alpha_b=args.prior_alpha_b,
            )
            est_model = HawkesExponential(est_model.mu, res.alpha, res.beta)
            print("MAP-EM-Exo: loglik=", res.loglik, " iters=", res.n_iter, " theta shape:", res.theta.shape)
        elif getattr(args, 'exo_joint', False):
            joint = fit_cox_hawkes_joint(
                train_events, T_train, args.dim, exo_i,
                init_theta=theta0, init_alpha=est_model.alpha, init_beta=est_model.beta,
                max_iter=args.exo_max_iter, step_theta=args.exo_step, step_alpha=args.step_alpha,
                step_beta=0.0 if getattr(args, 'freeze_beta', False) else args.step_beta, min_beta=args.min_beta, l2_alpha=args.l2_alpha, rho_max=args.rho_max,
            )
            est_model = HawkesExponential(joint.theta @ 0 + est_model.mu, joint.alpha, joint.beta)  # keep Hawkes for plotting
            print("Exogenous joint (theta,alpha,beta) fit: loglik=", joint.loglik, " iters=", joint.n_iter)
        else:
            exo_fit = fit_cox_hawkes_theta(train_events, T_train, args.dim, exo_i, est_model.alpha, est_model.beta, init_theta=theta0,
                                           step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                                           grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo)
            print("Exogenous baseline (Cox×Hawkes) theta fit:")
            print("theta shape:", exo_fit.theta.shape, " loglik:", exo_fit.loglik)
            if getattr(args, 'exo_em', False):
                emres = em_update_alpha(train_events, T_train, args.dim, exo_fit.theta, est_model.alpha, est_model.beta, max_iter=max(20, args.exo_max_iter // 10))
                est_model = HawkesExponential(est_model.mu, emres.alpha, emres.beta)
                print("EM alpha updated: iters=", emres.n_iter)
        # 在验证窗口上评估 full model（Cox×Hawkes）
        exo_val = build_proxy_exogenous(val_events, T_val, args.dim, window=args.exo_window, standardize=args.exo_standardize)
        exo_val_i = with_intercept(exo_val)
        from workflow.models.cox_hawkes import CoxHawkesExponential
        model_full = CoxHawkesExponential(exo_fit.theta if 'exo_fit' in locals() else theta0, est_model.alpha, est_model.beta, exo_val_i)
        ll_full = model_full.loglikelihood(val_events, T_val)
        comp_full = {
            'full_loglik': float(ll_full),
            'full_aic': float(2 * (exo_val.num_features * args.dim + est_model.alpha.size + est_model.beta.size) - 2 * ll_full),
        }
        print("Full model (Cox×Hawkes) validation metrics:", comp_full)
        if args.plot:
            if getattr(args, 'no_show', False):
                plot_exogenous_trajectories(exo_i.breakpoints, exo_i.features, savepath='docs/img/exo_features.png')
            else:
                plot_exogenous_trajectories(exo_i.breakpoints, exo_i.features)

    # compare with Poisson baseline
    print("Model comparison (AIC) on validation:", comp)

    # 先保存，再绘图（避免阻塞）
    if args.out:
        save_events_json(args.out, events)
        print(f"Events saved to {args.out}")

    if args.plot:
        grid = np.linspace(0, args.T, 500)
        intens = est_model.intensity_over_grid(events, grid)
        if getattr(args, 'no_show', False):
            plot_event_raster(events, dim=args.dim, T=args.T, title="Events and fitted model", savepath="fit_raster.png")
            plot_intensity(grid, intens, title="Fitted intensity", savepath="fit_intensity.png")
            resids = est_model.compensate_residuals(events, args.T)
            plot_residuals(resids, savepath="fit_residuals.png")
            plot_adjacency_heatmap(est_model.alpha, threshold=getattr(args, 'adj_threshold', 0.0), savepath="fit_adjacency.png")
            print("Saved: fit_raster.png, fit_intensity.png, fit_residuals.png, fit_adjacency.png")
        else:
            plot_event_raster(events, dim=args.dim, T=args.T, title="Events and fitted model")
            plot_intensity(grid, intens, title="Fitted intensity")
            resids = est_model.compensate_residuals(events, args.T)
            plot_residuals(resids)
            plot_adjacency_heatmap(est_model.alpha, threshold=getattr(args, 'adj_threshold', 0.0))


def main():
    parser = argparse.ArgumentParser(description="Hawkes modeling and evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="simulate Hawkes process")
    p_sim.add_argument("--dim", type=int, default=2)
    p_sim.add_argument("--T", type=float, default=10.0)
    p_sim.add_argument("--mu", type=float, default=0.2)
    p_sim.add_argument("--alpha", type=float, default=0.5)
    p_sim.add_argument("--beta", type=float, default=1.5)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--plot", action="store_true")
    p_sim.add_argument("--no_show", action="store_true", help="save figures only, no GUI window")
    p_sim.add_argument("--min_events", type=int, default=1, help="minimum number of events to simulate")
    p_sim.add_argument("--max_retries", type=int, default=10, help="max retries to extend T")
    p_sim.add_argument("--out", type=str, default=None, help="save simulated events to JSON path")
    p_sim.add_argument("--model", type=str, default="cox_hawkes", choices=["hawkes", "cox_hawkes"], help="simulation model")
    p_sim.add_argument("--period", type=float, default=30.0, help="period (s) for time exogenous in Cox×Hawkes simulation")
    p_sim.add_argument("--exo_step_sim", type=float, default=0.5, help="time grid step (s) for Cox×Hawkes time exogenous simulation")
    p_sim.add_argument("--sin_amp", type=float, default=1.0, help="amplitude on sin time feature in Cox×Hawkes simulation")
    p_sim.add_argument("--cos_amp", type=float, default=0.5, help="amplitude on cos time feature in Cox×Hawkes simulation")
    p_sim.set_defaults(func=run_simulate)

    p_fit = sub.add_parser("fit", help="fit Hawkes parameters and compare with Poisson; supports Cox×Hawkes")
    p_fit.add_argument("--dim", type=int, default=2)
    p_fit.add_argument("--T", type=float, default=10.0)
    p_fit.add_argument("--mu", type=float, default=0.2)
    p_fit.add_argument("--alpha", type=float, default=0.5)
    p_fit.add_argument("--beta", type=float, default=1.5)
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--plot", action="store_true")
    p_fit.add_argument("--no_show", action="store_true", help="save figures only, no GUI window")
    p_fit.add_argument("--input", type=str, default=None, help="load events from JSON; or 'bund' to load Bund demo")
    p_fit.add_argument("--bund_path", type=str, default=None, help="local tick-datasets root or full path to bund.npz")
    p_fit.add_argument("--model", type=str, default="hawkes", choices=["hawkes", "cox_hawkes"], help="model type")
    p_fit.add_argument("--use_exo", action="store_true", help="enable exogenous baseline (Cox×Hawkes)")
    p_fit.add_argument("--exo_window", type=float, default=1.0, help="window (s) for exogenous proxy features")
    p_fit.add_argument("--exo_step", type=float, default=1e-3, help="learning rate for theta")
    p_fit.add_argument("--exo_max_iter", type=int, default=300, help="iterations for theta or joint (theta,alpha,beta)")
    p_fit.add_argument("--exo_joint", action="store_true", help="jointly optimize theta and alpha/beta")
    p_fit.add_argument("--exo_standardize", action="store_true", help="standardize exogenous features")
    p_fit.add_argument("--exo_kind", type=str, default="proxy", choices=["proxy", "time"], help="type of exogenous features for Full")
    p_fit.add_argument("--period", type=float, default=30.0, help="period for time exogenous when exo_kind=time")
    p_fit.add_argument("--exo_step_sim", type=float, default=0.5, help="step for time exogenous when exo_kind=time")
    p_fit.add_argument("--grad_clip", type=float, default=10.0, help="gradient clipping for exo")
    p_fit.add_argument("--lr_decay_exo", type=float, default=0.0, help="learning rate decay for exo")
    p_fit.add_argument("--exo_em", action="store_true", help="after theta fit, EM update alpha")
    p_fit.add_argument("--method", type=str, default="mle", choices=["mle", "map_em", "map_em_exo", "tick_mle", "tick_full_em"], help="fitting method")
    p_fit.add_argument("--out", type=str, default=None, help="save events to JSON")
    # 稳定性相关可调参数
    p_fit.add_argument("--max_iter", type=int, default=600)
    p_fit.add_argument("--step_mu", type=float, default=1e-2)
    p_fit.add_argument("--step_alpha", type=float, default=1e-2)
    p_fit.add_argument("--step_beta", type=float, default=5e-4)
    p_fit.add_argument("--min_beta", type=float, default=0.1)
    p_fit.add_argument("--l2_alpha", type=float, default=0.0)
    p_fit.add_argument("--rho_max", type=float, default=0.95, help="spectral radius cap for stability")
    p_fit.add_argument("--adj_threshold", type=float, default=0.0, help="threshold for sparsifying adjacency")
    # MAP-EM 先验与选项
    p_fit.add_argument("--prior_mu_a", type=float, default=1.0)
    p_fit.add_argument("--prior_mu_b", type=float, default=1.0)
    p_fit.add_argument("--prior_alpha_a", type=float, default=1.0)
    p_fit.add_argument("--prior_alpha_b", type=float, default=1.0)
    p_fit.add_argument("--prior_beta_a", type=float, default=1.0)
    p_fit.add_argument("--prior_beta_b", type=float, default=1.0)
    p_fit.add_argument("--update_beta", action="store_true", help="whether to update beta in MAP-EM (default: freeze)")
    # 训练/验证窗口与冻结 beta
    p_fit.add_argument("--split_train", type=float, default=None, help="train proportion; split_train + split_val = 1")
    p_fit.add_argument("--split_val", type=float, default=None, help="validation proportion; split_train + split_val = 1")
    p_fit.add_argument("--freeze_beta", action="store_true", help="freeze beta (do not update in EM/MLE)")
    # tick 相关参数
    p_fit.add_argument("--tick_decays", type=float, default=None, help="decay rate for tick HawkesExpKern/HawkesEM; defaults to --beta")
    p_fit.add_argument("--tick_lasso", type=float, default=0.0, help="L1 penalty strength for tick (if >0)")
    p_fit.add_argument("--tick_max_iter", type=int, default=1000, help="tick optimizer iterations")
    p_fit.set_defaults(func=run_fit)

    # tune 子命令：在已有或仿真数据上做网格搜索
    p_tune = sub.add_parser("tune", help="grid search for stabilization and model selection")
    p_tune.add_argument("--dim", type=int, default=2)
    p_tune.add_argument("--T", type=float, default=10.0)
    p_tune.add_argument("--input", type=str, default=None, help="load events from JSON; if empty will simulate first")
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
    # 可选：验证窗口与 beta 网格固定
    p_tune.add_argument("--split_train", type=float, default=None)
    p_tune.add_argument("--split_val", type=float, default=None)
    p_tune.add_argument("--beta_grid", type=float, nargs='*', default=None, help="if provided, select fixed beta from grid on validation")
    p_tune.add_argument("--freeze_beta", action="store_true", help="freeze beta during grid (no updates)")

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
        # 训练/验证切分（若提供）
        use_split = args.split_train is not None and args.split_val is not None
        if use_split:
            if abs(args.split_train + args.split_val - 1.0) > 1e-6:
                raise ValueError("--split_train 与 --split_val 之和必须为 1.0")
            train_events, T_train, val_events, T_val = _split_and_shift_events(events, T, args.split_train, args.split_val)
        else:
            train_events, T_train = events, T
            val_events, T_val = events, T

        report = grid_search(
            train_events, T_train, args.dim,
            grid_min_beta=args.grid_min_beta,
            grid_l2_alpha=args.grid_l2_alpha,
            grid_rho_max=args.grid_rho_max,
            max_iter=args.max_iter,
            step_mu=args.step_mu,
            step_alpha=args.step_alpha,
            step_beta=(0.0 if args.freeze_beta else args.step_beta),
            val_events=val_events,
            T_val=T_val,
            beta_grid=args.beta_grid,
            freeze_beta=args.freeze_beta,
        )
        print("最优: ", report['best'])
        return report

    p_tune.set_defaults(func=run_tune)

    # gof 子命令：做三件套 GOF 检验并出图
    p_gof = sub.add_parser("gof", help="GOF tests: KS(Exp/Uniform) + Ljung-Box and QQ/hist plots")
    p_gof.add_argument("--dim", type=int, default=2)
    p_gof.add_argument("--T", type=float, default=30.0)
    p_gof.add_argument("--input", type=str, required=True, help="events JSON path")
    p_gof.add_argument("--method", type=str, default="mle", choices=["mle", "map_em"])
    p_gof.add_argument("--jitter", action="store_true", help="add small jitter to timestamps")
    p_gof.add_argument("--seasonal_bins", type=int, default=0, help=">0 to estimate piecewise-constant mu initial guess")
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
    p_gof.add_argument("--l2_alpha", type=float, default=0.0, help="L2 regularization on alpha (stability)")
    # 可选：外生因子基线（Cox×Hawkes）以增强GOF
    p_gof.add_argument("--use_exo", action="store_true", help="use Cox×Hawkes exogenous baseline for GOF")
    p_gof.add_argument("--exo_window", type=float, default=1.0, help="window (s) for exogenous proxy features")
    p_gof.add_argument("--exo_step", type=float, default=1e-3, help="learning rate for theta")
    p_gof.add_argument("--exo_max_iter", type=int, default=300, help="iterations for theta")
    p_gof.add_argument("--exo_standardize", action="store_true", help="standardize exogenous features")
    p_gof.add_argument("--grad_clip", type=float, default=10.0, help="gradient clipping for exo")
    p_gof.add_argument("--lr_decay_exo", type=float, default=0.0, help="learning rate decay for exo")
    # 训练/验证窗口与冻结 beta
    p_gof.add_argument("--split_train", type=float, default=None)
    p_gof.add_argument("--split_val", type=float, default=None)
    p_gof.add_argument("--freeze_beta", action="store_true")

    def run_gof(args: argparse.Namespace):
        # Lazy import to avoid requiring statsmodels unless GOF is used
        from workflow.gof import ks_exp_test, ks_uniform_test, ljung_box_test, compute_uniform_from_residuals
        from workflow.gof import plot_gof_hist, plot_gof_qq
        events = load_events_json(args.input)
        if args.jitter:
            events = add_jitter_to_events(events, eps=1e-6)
        # 切分
        use_split = args.split_train is not None and args.split_val is not None
        if use_split:
            if abs(args.split_train + args.split_val - 1.0) > 1e-6:
                raise ValueError("--split_train 与 --split_val 之和必须为 1.0")
            train_events, T_train, val_events, T_val = _split_and_shift_events(events, args.T, args.split_train, args.split_val)
        else:
            train_events, T_train = events, args.T
            val_events, T_val = events, args.T
        if args.seasonal_bins and args.seasonal_bins > 0:
            mu0 = estimate_piecewise_mu(events, args.T, args.dim, bins=args.seasonal_bins)
        else:
            mu0 = None
        # 拟合（先得到 alpha/beta 与常数基线 mu 或用于 exo 的初值）
        if args.method == 'map_em':
            res = map_em_exponential(
                train_events, T=T_train, dim=args.dim, init_mu=mu0,
                max_iter=args.max_iter, min_beta=args.min_beta,
                prior_mu_a=args.prior_mu_a, prior_mu_b=args.prior_mu_b,
                prior_alpha_a=args.prior_alpha_a, prior_alpha_b=args.prior_alpha_b,
                prior_beta_a=args.prior_beta_a, prior_beta_b=args.prior_beta_b,
                update_beta=(not getattr(args, 'freeze_beta', False)),
            )
            mu, alpha, beta = res.mu, res.alpha, res.beta
        else:
            fit = fit_hawkes_exponential(
                train_events, T=T_train, dim=args.dim, max_iter=args.max_iter,
                step_mu=args.step_mu, step_alpha=args.step_alpha, step_beta=(0.0 if getattr(args, 'freeze_beta', False) else args.step_beta),
                min_beta=args.min_beta, l2_alpha=args.l2_alpha, rho_max=args.rho_max,
            )
            mu, alpha, beta = fit.mu, fit.alpha, fit.beta
        # 若使用外生因子基线，则在固定 α/β 的前提下拟合 θ，并用 Cox×Hawkes 计算残差
        if getattr(args, 'use_exo', False):
            from workflow.features.exogenous import build_proxy_exogenous
            from workflow.fit.mle_exo import fit_cox_hawkes_theta
            from workflow.models.cox_hawkes import CoxHawkesExponential
            exo = build_proxy_exogenous(
                train_events, T_train, args.dim,
                window=args.exo_window, standardize=args.exo_standardize,
            )
            from workflow.features.exogenous import with_intercept
            exo_i = with_intercept(exo)
            theta0 = np.zeros((args.dim, exo_i.num_features))
            theta0[:, 0] = np.log(np.maximum(mu, 1e-8)) if isinstance(mu, np.ndarray) else 0.0
            exo_fit = fit_cox_hawkes_theta(
                train_events, T_train, args.dim, exo_i,
                alpha=alpha, beta=beta, init_theta=theta0,
                step=args.exo_step, max_iter=args.exo_max_iter, adam=True,
                grad_clip=args.grad_clip, lr_decay=args.lr_decay_exo,
            )
            exo_val = build_proxy_exogenous(val_events, T_val, args.dim, window=args.exo_window, standardize=args.exo_standardize)
            exo_val_i = with_intercept(exo_val)
            model = CoxHawkesExponential(exo_fit.theta, alpha, beta, exo_val_i)
        else:
            model = HawkesExponential(mu, alpha, beta)
        # 残差与GOF（验证窗口）
        resids = model.compensate_residuals(val_events, T_val)
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

    # exp 子命令：按题述步骤跑全流程（多种子），强制 tick + EM 与 tick MLE 对比
    p_exp = sub.add_parser("exp", help="End-to-end experiment: simulate ~100k events, fit Hawkes (tick) vs Full (tick+EM), GOF, and compare.")
    p_exp.add_argument("--dim", type=int, default=2)
    p_exp.add_argument("--mu", type=float, default=0.2)
    p_exp.add_argument("--alpha", type=float, default=0.5)
    p_exp.add_argument("--beta", type=float, default=1.5)
    p_exp.add_argument("--min_events", type=int, default=100000)
    p_exp.add_argument("--max_retries", type=int, default=10)
    p_exp.add_argument("--T", type=float, default=10.0)
    p_exp.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44])
    # simulation (time exogenous) controls for exp
    p_exp.add_argument("--model", type=str, default="cox_hawkes", choices=["hawkes", "cox_hawkes"])
    p_exp.add_argument("--period", type=float, default=30.0)
    p_exp.add_argument("--exo_step_sim", type=float, default=0.5)
    p_exp.add_argument("--sin_amp", type=float, default=1.0)
    p_exp.add_argument("--cos_amp", type=float, default=0.5)
    p_exp.add_argument("--exo_window", type=float, default=1.0)
    p_exp.add_argument("--exo_max_iter", type=int, default=300)
    p_exp.add_argument("--exo_step", type=float, default=1e-3)
    p_exp.add_argument("--exo_standardize", action="store_true")
    p_exp.add_argument("--tick_lasso", type=float, default=0.0)
    p_exp.add_argument("--tick_max_iter", type=int, default=1000)
    p_exp.add_argument("--beta_grid", type=float, nargs='+', default=[0.6, 1.0, 1.4], help="grid of fixed betas (decays) to select by validation AIC")
    p_exp.add_argument("--no_show", action="store_true")
    p_exp.add_argument("--plot", action="store_true")
    p_exp.add_argument("--split_train", type=float, default=0.8)
    p_exp.add_argument("--split_val", type=float, default=0.2)
    p_exp.add_argument("--out", type=str, default=None, help="save aggregated results JSON to path")
    p_exp.add_argument("--omp_threads", type=int, default=None, help="set OMP_NUM_THREADS (and MKL/NUMEXPR) for inner libs")
    p_exp.add_argument("--n_jobs", type=int, default=1, help="parallel jobs for seeds; beta_grid remains serial inside each job")

    def run_exp(args: argparse.Namespace):
        from workflow.gof import ks_exp_test, ks_uniform_test, ljung_box_test, compute_uniform_from_residuals
        from workflow.gof import plot_gof_hist, plot_gof_qq
        import json, os
        import multiprocessing as mp
        import math
        # control threading to avoid oversubscription when running grid or multiple seeds externally
        if args.omp_threads is not None and args.omp_threads > 0:
            os.environ['OMP_NUM_THREADS'] = str(int(args.omp_threads))
            os.environ['MKL_NUM_THREADS'] = str(int(args.omp_threads))
            os.environ['NUMEXPR_NUM_THREADS'] = str(int(args.omp_threads))
        # simulate per exp call (shared across workers)
        results = []
        for seed in args.seeds:
            # Step 1: simulate ~100k events
            sim_args = argparse.Namespace(
                dim=args.dim, T=args.T, seed=seed, mu=args.mu, alpha=args.alpha, beta=args.beta,
                plot=False, no_show=args.no_show, min_events=args.min_events, max_retries=args.max_retries, out=None,
                model=args.model, period=args.period, exo_step_sim=args.exo_step_sim, sin_amp=args.sin_amp, cos_amp=args.cos_amp
            )
            events, _ = run_simulate(sim_args)
            T_total = float(events[-1][0]) if events else args.T
            # Step 2/3: split 80/20
            train_events, T_train, val_events, T_val = _split_and_shift_events(events, T_total, args.split_train, args.split_val)
            # payload for worker
            payload = {
                'seed': seed,
                'dim': args.dim,
                'beta_grid': list(args.beta_grid),
                'tick_lasso': args.tick_lasso,
                'tick_max_iter': args.tick_max_iter,
                'exo_window': args.exo_window,
                'exo_step': args.exo_step,
                'exo_max_iter': args.exo_max_iter,
                'exo_standardize': args.exo_standardize,
                'exo_kind': 'time',
                'period': args.period,
                'exo_step_sim': args.exo_step_sim,
                'train_events': train_events,
                'val_events': val_events,
                'T_train': T_train,
                'T_val': T_val,
            }
            results.append(payload)

        # decide jobs: seeds-parallel if multiple seeds; if single seed, parallelize beta inside worker by reducing omp threads is suggested externally
        if isinstance(args.seeds, (list, tuple)) and len(args.seeds) > 1:
            cores = os.cpu_count() or 1
            base_threads = int(args.omp_threads) if args.omp_threads else 1
            max_jobs = max(1, cores // max(1, base_threads))
            effective_jobs = max(1, min(int(args.n_jobs), len(results), max_jobs))
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=effective_jobs) as pool:
                per_seed_results = pool.map(_exp_worker_seed, results)
        else:
            # single seed: run in-process; user can set a smaller omp_threads to avoid oversubscription
            per_seed_results = [_exp_worker_seed(results[0])]
        # Aggregate and conclude
        h_ll = np.array([r['hawkes']['ll'] for r in per_seed_results], dtype=float)
        h_aic = np.array([r['hawkes']['aic'] for r in per_seed_results], dtype=float)
        f_ll = np.array([r['full']['ll'] for r in per_seed_results], dtype=float)
        f_aic = np.array([r['full']['aic'] for r in per_seed_results], dtype=float)
        aggregate = {
            'Hawkes': {'LL': float(h_ll.mean()), 'LL_std': float(h_ll.std()), 'AIC': float(h_aic.mean()), 'AIC_std': float(h_aic.std())},
            'Full': {'LL': float(f_ll.mean()), 'LL_std': float(f_ll.std()), 'AIC': float(f_aic.mean()), 'AIC_std': float(f_aic.std())},
        }
        print("Aggregate (mean ± std) across seeds:")
        print(aggregate)
        better = 'Full' if f_aic.mean() < h_aic.mean() and f_ll.mean() > h_ll.mean() else 'Hawkes'
        print(f"Conclusion: {better} model performs better on validation by AIC/LL.")
        # save to JSON if requested (sanitize numpy types and large arrays)
        if args.out:
            def _to_serializable(obj):
                import numpy as _np
                if isinstance(obj, dict):
                    # drop large arrays such as 'U' from KS results
                    return {k: _to_serializable(v) for k, v in obj.items() if k != 'U'}
                if isinstance(obj, (list, tuple)):
                    return [_to_serializable(v) for v in obj]
                if isinstance(obj, (_np.generic,)):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    # avoid dumping huge arrays; keep small ones as lists
                    if obj.size > 1000:
                        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
                    return obj.tolist()
                return obj

            payload = {
                'seeds': list(args.seeds),
                'beta_grid': list(args.beta_grid),
                'per_seed': _to_serializable(per_seed_results),
                'aggregate': _to_serializable(aggregate),
                'conclusion': better,
            }
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return per_seed_results

    p_exp.set_defaults(func=run_exp)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
