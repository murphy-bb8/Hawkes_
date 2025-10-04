import pytest
import numpy as np

from workflow.data.bund import fetch_bund_events
from workflow.features.exogenous import build_proxy_exogenous
from workflow.fit.mle_exo import fit_cox_hawkes_theta


@pytest.mark.parametrize("day", [0, None])
def test_bund_fetch_and_exo(day):
    events, T = fetch_bund_events(day=day)
    assert isinstance(events, list)
    assert T >= 0.0
    dim = 4
    exo = build_proxy_exogenous(events, max(T, 1.0), dim=dim, window=2.0)
    assert exo.features.shape[0] == len(exo.breakpoints) - 1
    theta_fit = fit_cox_hawkes_theta(events, max(T, 1.0), dim, exo, alpha=np.ones((dim, dim)), beta=np.ones((dim, dim)), max_iter=5)
    assert theta_fit.theta.shape[0] == dim


