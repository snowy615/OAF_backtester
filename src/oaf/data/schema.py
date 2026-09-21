"""Canonical warehouse tables. Everything ingested is normalised to these shapes.

prices        one row per (date, ticker). ``close`` is the traded price;
              ``adj_close`` is split/dividend adjusted and drives returns.
              Delisted tickers stay in the table - their rows simply stop.
membership    point-in-time index/universe constituents as [start_date, end_date]
              intervals (end_date null = still a member). This is what stops a
              backtest from only ever seeing today's survivors.
fundamentals  long format with BOTH ``period_end`` (what the number describes) and
              ``available_date`` (when the market could first know it). Panels are
              built on ``available_date`` only.
meta          static per-ticker info plus delisting date/return.
"""

PRICES = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
PRICE_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]
MEMBERSHIP = ["universe", "ticker", "start_date", "end_date"]
FUNDAMENTALS = ["ticker", "field", "period_end", "available_date", "value"]
META = ["ticker", "name", "sector", "delisted_date", "delisting_return"]

TABLES = {
    "prices": PRICES,
    "membership": MEMBERSHIP,
    "fundamentals": FUNDAMENTALS,
    "meta": META,
}
