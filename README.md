# 使用 Hawkes 过程进行订单流建模：毒性识别与价格冲击预测

本项目实现了基于指数核的多维 Hawkes 过程，用于高频订单流建模；提供泊松过程基线、Ogata 薄化仿真、最大似然估计（MLE）、可视化与模型比较，并给出“毒性”市价单识别与短期价格冲击的简易度量。

## 安装

```bash
pip install -r requirements.txt
```

## 目录结构

- `workflow/`：按环节组织的代码包
  - `models/`：模型
    - `hawkes.py`：多维 Hawkes（指数核），仿真/似然/残差/强度
  - `fit/`：参数估计
    - `mle.py`：Hawkes 指数核的投影梯度 MLE
  - `baselines/`：基线模型
    - `poisson.py`：均匀泊松过程与率 MLE
  - `eval/`：模型比较
    - `compare.py`：Hawkes vs Poisson 的 AIC/似然
  - `viz/`：可视化
    - `plots.py`：事件栅格、强度曲线、残差直方图（中文友好）
  - `io/`：数据读写
    - `events.py`：事件 JSON 读写
  - `tuning/`：调参与稳定性
    - `runner.py`：多随机种子的仿真-拟合-比较-谱半径
  - `analytics/`：交易分析
    - `toxicity.py`：市价单“毒性”评分与短期价格冲击代理
  - `tick_integration/`：tick 可选适配
    - `adapter.py`：使用 tick.hawkes 的拟合接口（未安装会跳过）
- `main.py`：命令行入口（仿真/拟合/保存加载）

## 快速开始

仿真（保存JSON与图片，不弹窗）：

```bash
python main.py simulate --dim 1 --T 30 --mu 0.6 --alpha 0.7 --beta 1.2 --plot --no_show --min_events 60 --max_retries 80 --out events.json
```

MLE 或 MAP-EM 拟合，并与泊松基线比较：

```bash
# 稳定化 MLE
python main.py fit --dim 1 --T 30 --input events.json --plot --no_show \
  --method mle --max_iter 3000 --step_mu 5e-3 --step_alpha 5e-3 --step_beta 1e-4 \
  --min_beta 0.4 --rho_max 0.85

# MAP-EM（带先验）
python main.py fit --dim 1 --T 30 --input events.json --plot --no_show \
  --method map_em --max_iter 500 --min_beta 0.4 \
  --prior_mu_a 1.0 --prior_mu_b 1.0 --prior_alpha_a 1.0 --prior_alpha_b 1.0 \
  --prior_beta_a 2.0 --prior_beta_b 2.0
```

## 调参与稳定性（tuning）

使用 `workflow/tuning/runner.py` 在多个随机种子下重复“仿真-拟合-比较”，并输出谱半径（`alpha/beta` 的谱半径<1 表示稳定）：

```python
from workflow.tuning import simulate_and_tune
report = simulate_and_tune(dim=1, T=30.0, seeds=(1,2,3), mu=0.6, alpha=0.7, beta=1.2)
print(report)

## 拟合优度（三件套）

- KS 对 Exp(1)、KS 对 Uniform（对 U=1-exp(-ΔΛ)）与 Ljung–Box/ACF 自相关检验
- p 值 >= 0.05 视为“不过度拒绝”（可用 0.01 为强标准）
- 输出：样本量 n、KS_Exp D/p、KS_Uni D/p、LB Q/p；并保存 QQ-plot 与直方图

```bash
python main.py gof --dim 1 --T 30 --input events.json --method mle --jitter --seasonal_bins 10
```

说明：
- --jitter 会对秒/毫秒时间戳加 1e-6 抖动，消除同刻堆叠；
- --seasonal_bins>0 会用分段常数估计基线初值，吸收盘中季节性（U 型/收盘拥堵）。


## 毒性识别与价格冲击

`analytics.py` 提供：
- 毒性分数：短期窗口内由该笔市价单引发的期望“子事件”增量的近似值；
- 冲击代理：价格冲击可取毒性分数的线性函数（系数可用实证估计）。

## 进阶：结合 tick 包与本地 MHP 类

- 可选方案：使用 `tick` 包（`tick.hawkes`）进行参数估计/仿真，以对比本实现（需 `pip install tick`）。
- 也可沿用本仓库 `MHP`/`HawkesExponential`，前者适合入门演示，后者支持多维与评估工具链。

## 方法说明

- 仿真：Ogata 薄化；
- 似然：指数核的闭式积分；
- 估计：投影梯度上升（非负约束）；
- 诊断：时间重标定残差应近似 Exp(1)。

## 备注

- 稳定性：`alpha/beta` 的谱半径<1；
- 多维：维度可映射买/卖/不同事件类型；
- 可扩展：替换核函数、加入外生因子或价格过程联立建模。
