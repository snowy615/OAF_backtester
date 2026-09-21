import numpy as np
import pandas as pd
import pytest

from oaf.data.synthetic import write_demo_warehouse
from oaf.panel import Panel
from oaf.spec import Construction, Param, StrategySpec, Universe


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    return write_demo_warehouse(tmp_path_factory.mktemp("wh"), n_tickers=40, n_days=900, seed=3)


@pytest.fixture(scope="session")
def panel(warehouse):
    return warehouse.load_panel("demo_index")


@pytest.fixture
def momentum_spec():
    return StrategySpec(
        name="mom",
        signal="rank(ts_delay(close, skip) / ts_delay(close, skip + lookback) - 1)",
        universe=Universe(name="demo_index"),
        params=[
            Param(name="lookback", value=60, low=20, high=120, kind="int"),
            Param(name="skip", value=5, low=1, high=10, kind="int"),
            Param(name="q", value=0.2, low=0.1, high=0.3),
        ],
        construction=Construction(long_quantile="q", short_quantile="q", max_weight="0.2"),
        rebalance="weekly",
    )


def make_panel(close: pd.DataFrame, mask: pd.DataFrame | None = None) -> Panel:
    """Tiny hand-built panel for exact-arithmetic tests."""
    returns = close.pct_change(fill_method=None)
    mask = close.notna() if mask is None else mask
    return Panel(fields={"close": close, "returns": returns}, mask=mask)


@pytest.fixture
def toy_panel():
    dates = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 6)), axis=0)), index=dates, columns=list("ABCDEF")
    )
    return make_panel(close)
