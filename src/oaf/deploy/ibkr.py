"""Interactive Brokers execution (TWS or IB Gateway, via ``ib_async``).

Year 1 is paper-only, and the code enforces it rather than trusting the operator:

- ``mode="paper"`` refuses the well-known live ports and refuses any account whose id
  does not look like an IBKR paper account (``D...``).
- ``mode="live"`` is locked unless the environment opts in with
  ``OAF_ENABLE_LIVE_TRADING=YES-I-UNDERSTAND`` *and* the caller passes ``allow_live``.
- Nothing is sent unless ``send=True``; the default is a dry run that prints the orders.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from .orders import DeploymentPlan, Order

LIVE_PORTS = {7496, 4001}
PAPER_PORTS = {7497, 4002}
LIVE_ENV, LIVE_TOKEN = "OAF_ENABLE_LIVE_TRADING", "YES-I-UNDERSTAND"


class DeploymentBlocked(RuntimeError):
    """A safety gate refused the deployment."""


@dataclass
class IBKRConfig:
    mode: Literal["paper", "live"] = "paper"
    host: str = field(default_factory=lambda: os.environ.get("OAF_IBKR_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("OAF_IBKR_PORT", "7497")))
    client_id: int = field(default_factory=lambda: int(os.environ.get("OAF_IBKR_CLIENT_ID", "17")))
    account: Optional[str] = None  # required when the login manages several accounts
    currency: str = "USD"
    exchange: str = "SMART"
    order_type: Literal["MKT", "MOC"] = "MOC"  # the simulator assumes fills at the close
    allow_live: bool = False
    # per-ticker contract details where the default does not resolve, e.g.
    # {"VOD": {"exchange": "LSE", "currency": "GBP"}}
    contract_overrides: dict[str, dict] = field(default_factory=dict)


def check_gates(cfg: IBKRConfig) -> None:
    """Static safety checks, before any connection is made."""
    if cfg.mode == "paper":
        if cfg.port in LIVE_PORTS:
            raise DeploymentBlocked(f"port {cfg.port} is a live-trading port; paper mode uses {sorted(PAPER_PORTS)}")
        return
    if not cfg.allow_live or os.environ.get(LIVE_ENV) != LIVE_TOKEN:
        raise DeploymentBlocked(
            "live trading is locked (Year 1 is paper-only). To unlock in Year 2, set "
            f"{LIVE_ENV}={LIVE_TOKEN} and pass --live."
        )


class IBKRBroker:
    def __init__(self, cfg: IBKRConfig):
        check_gates(cfg)
        try:
            import ib_async
        except ImportError as e:
            raise ImportError("IBKR deployment needs ib_async: pip install 'oaf-backtester[ibkr]'") from e
        self.cfg = cfg
        self._ib_async = ib_async
        self.ib = ib_async.IB()
        self.account: Optional[str] = None

    def __enter__(self) -> "IBKRBroker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.ib.disconnect()

    def connect(self) -> None:
        self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
        accounts = list(self.ib.managedAccounts())
        account = self.cfg.account or (accounts[0] if len(accounts) == 1 else None)
        if account is None or account not in accounts:
            self.ib.disconnect()
            raise DeploymentBlocked(f"set IBKRConfig.account to one of {accounts}")
        if self.cfg.mode == "paper" and not account.upper().startswith("D"):
            self.ib.disconnect()
            raise DeploymentBlocked(f"account {account} is not a paper account (paper ids start with 'D')")
        self.account = account

    def positions(self) -> dict[str, int]:
        """Current stock positions as ticker -> signed share count."""
        return {
            p.contract.symbol: int(p.position)
            for p in self.ib.positions(self.account)
            if p.contract.secType == "STK" and p.position
        }

    def _contract(self, ticker: str):
        o = self.cfg.contract_overrides.get(ticker, {})
        return self._ib_async.Stock(
            o.get("symbol", ticker), o.get("exchange", self.cfg.exchange), o.get("currency", self.cfg.currency)
        )

    def place(self, orders: list[Order]) -> list[str]:
        """Send the orders. Returns one status line per order."""
        log = []
        for o in orders:
            contract = self._contract(o.ticker)
            if not self.ib.qualifyContracts(contract):
                log.append(f"SKIPPED {o.ticker}: IBKR could not resolve the contract")
                continue
            order = self._ib_async.Order(
                action=o.side, totalQuantity=o.quantity, orderType=self.cfg.order_type, account=self.account, tif="DAY"
            )
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.2)
            log.append(f"SENT {o.side} {o.quantity} {o.ticker} ({self.cfg.order_type}) -> {trade.orderStatus.status}")
        return log


def execute_plan(plan: DeploymentPlan, cfg: IBKRConfig, send: bool = False) -> list[str]:
    """Dry-run by default; with ``send=True`` the plan's orders go to IBKR."""
    if plan.mode != cfg.mode:
        raise DeploymentBlocked(f"plan was built for {plan.mode} but the broker config is {cfg.mode}")
    check_gates(cfg)
    if not send:
        return [f"DRY RUN {o.side} {o.quantity} {o.ticker} ~{o.est_notional:,.0f}" for o in plan.orders]
    with IBKRBroker(cfg) as broker:
        return broker.place(plan.orders)
