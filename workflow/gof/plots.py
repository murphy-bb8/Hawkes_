import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def _setup_cn():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def plot_gof_hist(residuals: List[float], savepath: Optional[str] = None):
    _setup_cn()
    arr = np.asarray(residuals, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(arr, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    xs = np.linspace(0, max(arr.max(), 1), 200)
    plt.plot(xs, np.exp(-xs), 'r--', label='Exp(1)')
    plt.xlabel('补偿残差')
    plt.ylabel('密度')
    plt.title('GOF: 残差直方图 vs Exp(1)')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


def plot_gof_qq(residuals: List[float], savepath: Optional[str] = None):
    _setup_cn()
    arr = np.asarray(residuals, dtype=float)
    arr = np.sort(arr)
    n = arr.size
    if n == 0:
        return
    # Theoretical quantiles for Exp(1): F^{-1}(p) = -log(1-p)
    ps = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log(1.0 - ps)
    plt.figure(figsize=(5, 5))
    plt.plot(theo, arr, 'o', ms=3)
    m = max(theo.max(), arr.max())
    plt.plot([0, m], [0, m], 'r--', lw=1)
    plt.xlabel('理论分位（Exp(1)）')
    plt.ylabel('样本分位（残差）')
    plt.title('GOF: QQ-plot 对 Exp(1)')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


