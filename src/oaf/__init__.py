"""Oxford Alpha Fund backtester: idea -> signal -> simulated portfolio -> metrics -> IBKR."""

from .panel import Panel
from .sim import BacktestResult, SimConfig, run_backtest
from .spec import Construction, Param, Rules, StrategySpec, Universe

__version__ = "0.1.0"
__all__ = [
    "Panel",
    "BacktestResult",
    "SimConfig",
    "run_backtest",
    "StrategySpec",
    "Param",
    "Universe",
    "Rules",
    "Construction",
]
