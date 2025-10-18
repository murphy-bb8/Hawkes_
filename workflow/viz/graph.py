import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def _setup_cn():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False


def plot_adjacency_heatmap(alpha: np.ndarray,
                           labels: Optional[List[str]] = None,
                           threshold: float = 0.0,
                           title: str = "Adjacency (Alpha)",
                           savepath: Optional[str] = None):
    _setup_cn()
    A = np.array(alpha, dtype=float)
    if threshold is not None and threshold > 0:
        A = np.where(A >= threshold, A, 0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap='Reds')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = A.shape[0]
    if labels is None:
        labels = [f"dim{i}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel('source j')
    ax.set_ylabel('target i')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


