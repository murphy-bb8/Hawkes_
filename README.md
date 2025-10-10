# 多维 Hawkes 过程（指数核）：高频订单流建模与评估工具集

本项目提供基于指数核的多维 Hawkes 过程建模与评估的完整工具链，覆盖：仿真、参数估计（MLE / MAP‑EM）、稳定性与正则化、外生因子（Cox×Hawkes）基线、拟合优度（KS/QQ/Ljung‑Box）与可视化。适用于订单流/逐笔成交等不齐次点过程数据的聚集性与互激励分析。

## 安装

```bash
pip install -r requirements.txt
```

## 目录结构

- `workflow/`：按环节组织的代码包
  - `models/`
    - `hawkes.py`：多维 Hawkes（指数核），仿真/似然/残差/强度
    - `legacy.py`：简化版单变量 Hawkes（入门/对照）
  - `fit/`
    - `mle.py`：投影梯度 MLE（非负约束、min_beta、谱半径投影 rho_max、可选 L2）
    - `map_em.py`：MAP-EM（Gamma 先验，支持 min_beta，默认不更新 beta）
  - `baselines/`
    - `poisson.py`：均匀泊松过程与率 MLE
  - `eval/`
    - `compare.py`：Hawkes vs Poisson 的 AIC/似然
  - `viz/`
    - `plots.py`：事件栅格、强度曲线、残差直方图
    - `graph.py`：稀疏传染图（Alpha 热力图，支持阈值显示）
  - `gof/`
    - `tests.py`：KS(Exp/Uniform)、Ljung–Box/ACF、Uniform 变换
    - `plots.py`：QQ-plot 与直方图
  - `preprocess/`
    - `jitter.py`：时间戳抖动（1e-6，消除同刻堆叠）
    - `seasonal.py`：分段常数基线初值估计（吸收盘中季节性）
  - `io/`
    - `events.py`：事件 JSON 读写
  - `tuning/`
    - `runner.py`：多随机种子的仿真-拟合-比较-谱半径
    - `grid.py`：min_beta / l2_alpha / rho_max 的网格搜索（AIC+残差指标）
  - `analytics/`
    - `toxicity.py`：市价单“毒性”评分与短期价格冲击代理
  - `tick_integration/`
    - `adapter.py`：tick.hawkes 的拟合接口（自动回退与清洗）
- `main.py`：命令行入口（仿真/拟合/保存加载）

## 快速开始

仿真（保存JSON与图片）：

```bash
python main.py simulate --dim 1 --T 30 --mu 0.6 --alpha 0.7 --beta 1.2 --plot --no_show --min_events 60 --max_retries 80 --out events.json
```

MLE 或 MAP-EM 拟合，并与基线比较：

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

## 初次实验结果（单维示例，T=30，n≈70）

- 参数估计（MLE，稳定化）：
  - mu ≈ 0.7402，alpha ≈ 0.8616，beta ≈ 1.2366，分枝比 G ≈ 0.70 < 1（稳定）
  - 对数似然 loglik ≈ 7.2776
  - AIC（Hawkes）≈ -8.5553，对比 Poisson AIC ≈ 23.3783（显著更优）
- 三件套 GOF（残差补偿检验）：
  - KS-Exp: D ≈ 0.0913，p ≈ 0.5721（不过度拒绝）
  - KS-Uniform: D ≈ 0.0913，p ≈ 0.5721（不过度拒绝）
  - Ljung–Box(20): Q ≈ 16.124，p ≈ 0.7089（无显著自相关）
- 结论：
  - 模型在该数据上拟合良好，较泊松基线有显著性能提升，残差分布与独立性检验均通过。

## 本次实验（2025-10-07，dim=4，T≈50391.74）

在 Bund 事件样本（dim=4，T≈50391.74）上运行 MAP-EM + 季节性基线初值与抖动，命令：

```bash
python main.py gof --dim 4 --T 50391.74 --input events.json \
  --method map_em --jitter --seasonal_bins 20 \
  --max_iter 800 --min_beta 0.5 --rho_max 0.85
```

输出图片：

- 事件与拟合模型：![fit_raster](docs/img/fit_raster_2025-10-07.png)
- 拟合强度轨迹：![fit_intensity](docs/img/fit_intensity_2025-10-07.png)
- 残差直方图：![fit_residuals](docs/img/fit_residuals_2025-10-07.png)
- 稀疏传染图：![fit_adjacency](docs/img/fit_adjacency_2025-10-07.png)

### 关键观察
- 维度2、维度3自激发较强，跨维度传染非对称；
- 约 3e4 时间点附近出现多维同步性峰值；
- 残差较 Exp(1) 有偏离，建议引入外生因子（`--use_exo`）或使用多核（双指数/幂律）进一步提升拟合度。
- 受数据爆发与不平衡影响，GOF 显著右尾。

### 改进策略

可操作改进（优先级从高到中）：
1) 启用外生因子（Cox×Hawkes 基线）吸收宏观/微观时变驱动
```bash
python main.py fit --input bund --dim 4 --T 0 --model cox_hawkes --use_exo --exo_standardize \
  --exo_window 2.0 --exo_step 1e-3 --exo_max_iter 500 --grad_clip 10 \
  --plot --no_show --method mle --max_iter 800 --step_alpha 5e-3 --step_beta 2e-4 \
  --min_beta 0.5 --rho_max 0.85
```
2) 稳定化与稀疏化：限制谱半径与对 `alpha` 加 L2/L1（后者可借助 tick 对照）
```bash
python main.py fit --input events.json --dim 4 --T 50391.74 --plot --no_show \
  --method mle --max_iter 1500 --step_alpha 5e-3 --step_beta 2e-4 \
  --min_beta 0.5 --rho_max 0.85 --l2_alpha 0.02
```
3) GOF 工作流
```bash
python main.py gof --dim 4 --T 50391.74 --input events.json \
  --method map_em --jitter --seasonal_bins 20 --max_iter 800 \
  --min_beta 0.5 --rho_max 0.85
```

说明：本次实验已是“多变量 Hawkes”（dim=4）。`--seasonal_bins 20` 仅用于估计基线初值（piecewise 常数），最终模型仍为常数基线的 Hawkes；若需真正的时变基线或外生驱动，应使用 Cox×Hawkes。

---

### 外生因子（Cox×Hawkes）
- 建模：`λ_i(t) = exp(θ_i^T X(t)) + Σ_j α_{ij} e^{-β_{ij}(t-t_k^j)}`，其中 `X(t)` 采用分段常数特征
- 特征：支持基于事件的 proxy（`flow+`/`flow-`/`rv`），或替换为真实 LOB/行情因子
- 稳定性：对 `θ^T X` 做裁剪避免指数溢出（logits ∈ [-50,50]），并支持特征标准化（均值-方差）
- 优化：
  - θ：Adam（带梯度裁剪、学习率衰减）
  - θ,α,β 联合：θ 用 Adam，α/β 加非负/下界与谱半径投影（ρ≤rho_max）
  - 可选 EM：在 θ 固定后对 α 做责任分配更新

## 基线建模（外生驱动）

- 已支持 Cox×Hawkes 外生基线：`λ_i(t) = exp(θ_i^T X(t)) + Σ_j α_{ij} e^{-β_{ij}(t-t_k^j)}`，`X(t)` 为分段常数特征；`θ` 由 MLE/Adam 学习。
- GOF 增强：`--use_exo` 会先用 MLE 得到 `α/β`，再拟合 `θ`，并用 Cox×Hawkes 计算残差做 KS/QQ/LB。
- 外生特征建议：OFI/QI、签名成交量/笔数、收益与波动率、买卖价差、成交密度、开收盘/公告时段 dummy 等。可通过 `workflow/features/exogenous.py` 扩展。

开启 Cox×Hawkes 与外生特征的推荐命令见上节“改进策略”。若需更强的稳健性，可在 θ 固定后执行 `--exo_em` 以 EM 更新 α，或使用 `--exo_joint` 联合优化 θ 与 α/β。


## 方法说明

- **过程与核**：指数核 Hawkes 过程，强度函数为：
  ```
  λᵢ(t) = μᵢ + Σⱼ Σₖ αᵢⱼ e^(-βᵢⱼ(t-tₖʲ))
  ```
  其中 μᵢ 为基线强度，αᵢⱼ 为激励强度，βᵢⱼ 为衰减参数。

- **仿真**：Ogata 薄化算法（自适应上界），时间复杂度接近 O(n)。

- **似然计算**：指数核的闭式对数似然与积分项，避免 O(n²) 逐对累加带来的数值下溢。

- **稳定性约束**：
  - 硬约束：β ≥ min_beta、谱半径投影 ρ(α/β) ≤ rho_max
  - 可选软约束：对 α 加 L2 正则化抑制过大分枝比（提升可解释性）

- **参数估计**：
  - **MLE**：投影梯度上升（步长可调、支持小步长+高迭代），对多维/多日可并行
  - **MAP-EM**：Gamma 先验（μ, α, β），E 步计算父子责任、M 步闭式更新（默认固定 β）

- **残差与拟合优度**：时间重标定残差 ΔΛ，进行 KS(Exp/Uniform) 与 Ljung–Box 检验；QQ/直方图核对 Exp(1) 分布。
- **季节性处理**：分段常数基线初值（`--seasonal_bins`）与时间戳抖动（`--jitter`）吸收盘中 U 型等模式。

- **稀疏传染图**：`--adj_threshold` 直观展示谁激励谁，结合 tick 的 L1 正则化可做对照。


CLI（Bund + exo 示例）
```powershell
# θ 拟合（标准化 + Adam + 梯度裁剪）
python .\main.py fit --input bund --dim 4 --T 0 --model cox_hawkes --use_exo --exo_standardize \
  --exo_window 2.0 --exo_step 1e-3 --lr_decay_exo 1e-4 --grad_clip 10 --exo_max_iter 500 \
  --plot --no_show --method mle --max_iter 1200 --step_mu 5e-3 --step_alpha 5e-3 --step_beta 2e-4 --min_beta 0.4 --rho_max 0.9

# θ,α,β 联合优化（更稳健参数）
python .\main.py fit --input bund --dim 4 --T 0 --model cox_hawkes --use_exo --exo_standardize --exo_joint \
  --exo_window 1.0 --exo_step 5e-4 --exo_max_iter 800 --grad_clip 10 --lr_decay_exo 1e-4 \
  --plot --no_show --method mle --max_iter 0 --step_alpha 2e-3 --step_beta 1e-4 --min_beta 0.6 --rho_max 0.85 --l2_alpha 0.02

# θ 后 EM 更新 α
python .\main.py fit --input bund --dim 4 --T 0 --model cox_hawkes --use_exo --exo_standardize --exo_em \
  --exo_window 2.0 --exo_step 1e-3 --exo_max_iter 500 --plot --no_show --method mle --max_iter 1200 \
  --step_mu 5e-3 --step_alpha 5e-3 --step_beta 2e-4 --min_beta 0.4 --rho_max 0.9
```

参考：Bund 数据来自 tick 的公开示例（20 日 × 4 通道）；外生因子稳健性、优化与标准化处理参考实务做法与相关开源实现（如 Kramer 的外生因子模型实现）。

## 扩展方向

- **稳定性保证**：`alpha/beta` 的谱半径 < 1；必要时提高 `min_beta`、降低 `rho_max`，并对 `alpha` 加 L2/L1 约束以稀疏化。
- **多维建模（已启用）**：当前为多变量 Hawkes（dim=4）。可进一步拆分维度（买/卖/不同触发类型）、或合并跨市场/跨合约事件。
- **核函数扩展**：双指数/幂律/非参数核，以刻画短记忆+长记忆；或采用可学习核（如分段常数核）。
- **时变基线**：从常数 μ 过渡到 Cox×Hawkes（`exp(θ^T X)`）或分段常数/样条基线，吸收宏观/季节性驱动。
- **估计稳健化**：引入早停、学习率衰减、梯度裁剪、谱半径投影；并提供 tick 的 L1 解作对照。
- **评估强化**：在 KS/QQ/LB 基础上，补充时间块交叉验证、留出日滚动评估与事后事件预测精度。

## 毒性识别与价格冲击（暂定）

`analytics.py` 提供：
- 毒性分数：短期窗口内由该笔市价单引发的期望“子事件”增量的近似值；
- 冲击代理：价格冲击可取毒性分数的线性函数（系数可用实证估计）。

## 备注

- 默认图像输出位于根目录或 `docs/exp_*/`；使用 `--no_show` 以避免交互阻塞。
- Windows/PowerShell 与 Linux 的路径分隔符不同，命令示例已统一为跨平台写法（相对路径不含反斜杠）。
- 若发生极端峰值与残差偏离，优先检查：时间戳抖动、分段季节性、谱半径约束与正则化是否打开；其次再引入外生特征或核函数扩展。
