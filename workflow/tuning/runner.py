import numpy as np
from typing import Dict, Any

from ..models.hawkes import HawkesExponential
from ..fit import fit_hawkes_exponential
from ..eval import compare_hawkes_poisson


def simulate_and_tune(dim: int = 1, T: float = 10.0, seeds=(1, 2, 3),
                      mu=0.2, alpha=0.5, beta=1.5) -> Dict[str, Any]:
    results = []
    for s in seeds:
        mu_vec = np.full(dim, mu)
        a = np.full((dim, dim), alpha)
        b = np.full((dim, dim), beta)
        model = HawkesExponential(mu_vec, a, b)
        events = model.simulate_ogata(T=T, seed=s)
        fit = fit_hawkes_exponential(events, T=T, dim=dim)
        comp = compare_hawkes_poisson(events, T, fit.mu, fit.alpha, fit.beta)
        results.append({
            'seed': s,
            'n_events': len(events),
            'fit_loglik': fit.loglik,
            'hawkes_aic': comp['hawkes_aic'],
            'poisson_aic': comp['poisson_aic'],
            'mu': fit.mu.tolist(),
            'alpha': fit.alpha.tolist(),
            'beta': fit.beta.tolist(),
        })
    from numpy.linalg import eigvals
    rho = []
    for r in results:
        alpha_arr = np.array(r['alpha'])
        beta_arr = np.array(r['beta'])
        G = alpha_arr / beta_arr
        rho.append(max(abs(eigvals(G))))
    return {'runs': results, 'spectral_radius': [float(x) for x in rho]}


