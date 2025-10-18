import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def _setup_chinese_fonts():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False


def plot_event_raster(events: List[Tuple[float, int]], dim: int, T: float, title: str = "Event Raster", savepath: Optional[str] = None):
    _setup_chinese_fonts()
    events = sorted(events, key=lambda x: x[0])
    plt.figure(figsize=(10, 0.8 * max(2, dim)))
    if len(events) == 0:
        plt.text(T * 0.5, dim * 0.5, "No events", ha='center', va='center', fontsize=14, color='gray')
    else:
        for t, i in events:
            plt.vlines(t, i + 0.1, i + 0.9, color='k', linewidth=1)
    plt.yticks(range(dim), [f"type {i}" for i in range(dim)])
    plt.xlim(0, T)
    plt.ylim(-0.1, dim + 0.1)
    plt.xlabel('time')
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


def plot_intensity(grid: np.ndarray, intensities: np.ndarray, labels: Optional[List[str]] = None, title: str = "Intensity", savepath: Optional[str] = None):
    _setup_chinese_fonts()
    plt.figure(figsize=(10, 4))
    for i in range(intensities.shape[1]):
        lab = labels[i] if labels and i < len(labels) else f"dim {i}"
        plt.plot(grid, intensities[:, i], label=lab)
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


def plot_residuals(residuals: List[float], title: str = "Residuals histogram (approx Exp(1))", savepath: Optional[str] = None):
    _setup_chinese_fonts()
    if len(residuals) == 0:
        print("No residuals to plot")
        return
    arr = np.asarray(residuals, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(arr, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('residual value')
    plt.ylabel('density')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    else:
        plt.show()


