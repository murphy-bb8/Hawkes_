import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_exogenous_trajectories(breakpoints: np.ndarray, features: np.ndarray, names: Optional[list] = None,
                                savepath: Optional[str] = None):
    t = breakpoints[:-1]
    k = features.shape[1]
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.2 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for j in range(k):
        axes[j].step(t, features[:, j], where='post')
        axes[j].set_ylabel(names[j] if names and j < len(names) else f"x{j}")
    axes[-1].set_xlabel('time')
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        plt.close(fig)
    else:
        plt.show()


