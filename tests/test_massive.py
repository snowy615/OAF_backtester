"""Massive client against a scripted transport (no network)."""

import urllib.parse

import pandas as pd
import pytest

from oaf.data import massive as mx
from oaf.data.universes import load_membership

DAY = 86_400_000
T0 = 1704153600000  # 2024-01-02 UTC


def _bar(t, c, sym=None):
    d = {"t": t, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 1000}
    if sym:
        d["T"] = sym
    return d


def transport(url):
    path, _, q = url.partition("?")
    qs = {k: v[0] for k, v in urllib.parse.parse_qs(q).items()}
    if "/v2/aggs/ticker/" in path:
        t = path.split("/")[6]
        assert qs["adjusted"] == "false"
        base = 100 if t == "AAPL" else 20
        return {"status": "OK", "results": [_bar(T0, base), _bar(T0 + DAY, base / 2 if t == "AAPL" else base)]}
    if "/v2/aggs/grouped/" in path:
        return {"status": "OK", "results": [_bar(T0, 100, "AAPL"), _bar(T0, 20, "ZZZ"), _bar(T0, 5, "SPY")]}
    if "/stocks/v1/splits" in path:
        return {"status": "OK", "results": [{"ticker": "AAPL", "execution_date": "2024-01-03", "split_from": 1, "split_to": 2, "adjustment_type": "forward_split"}]}
    if path.endswith("/v3/reference/tickers"):
        if qs.get("cursor"):
            return {"status": "OK", "results": [{"ticker": "ZZZ", "name": "Z", "primary_exchange": "XNYS", "type": "CS"}]}
        if qs["active"] == "false":
            return {"status": "OK", "results": [{"ticker": "DEAD", "name": "Gone", "delisted_utc": "2023-05-01T00:00:00Z", "primary_exchange": "XNYS", "type": "CS"}]}
        return {"status": "OK", "results": [{"ticker": "AAPL", "name": "Apple", "primary_exchange": "XNAS", "type": "CS"}],
                "next_url": "https://api.massive.com/v3/reference/tickers?cursor=x"}
    if "income-statements" in path:
        return {"status": "OK", "results": [{"tickers": ["AAPL"], "filing_date": "2024-02-02", "period_end": "2023-12-30", "revenue": 1e9, "basic_earnings_per_share": 2.1}]}
    if "balance-sheets" in path:
        return {"status": "OK", "results": [{"tickers": ["AAPL"], "filing_date": "2024-02-02", "period_end": "2023-12-30", "total_equity": 5e8}]}
    raise AssertionError(url)


@pytest.fixture
def client():
    return mx.MassiveClient(transport=transport)


def test_requires_a_key_without_a_transport(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(mx.MassiveError, match="MASSIVE_API_KEY"):
        mx.MassiveClient()


def test_daily_bars_keep_raw_close_and_rebuild_adjusted(client):
    px = mx.fetch_daily_bars(client, ["AAPL", "MSFT"], "2024-01-01", "2024-01-05").set_index(["ticker", "date"])
    d0, d1 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    assert px.loc[("AAPL", d0), "close"] == 100 and px.loc[("AAPL", d0), "adj_close"] == 50  # pre-split row halved
    assert px.loc[("AAPL", d1), "adj_close"] == 50  # no -50% return across the split
    assert px.loc[("MSFT", d0), "adj_close"] == 20
    assert list(px.reset_index().columns) == ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def test_market_fetch_filters_tickers_and_pages_tickers(client):
    meta = mx.fetch_tickers(client)
    assert meta["ticker"].tolist() == ["AAPL", "ZZZ", "DEAD"]  # paged + delisted list
    assert meta.set_index("ticker").loc["DEAD", "delisted_date"] == pd.Timestamp("2023-05-01")
    keep = set(meta["ticker"])
    px = mx.fetch_market(client, "2024-01-02", "2024-01-02", keep=keep.__contains__)
    assert sorted(px["ticker"]) == ["AAPL", "ZZZ"]  # SPY is not a common stock


def test_readjust_history_after_a_late_split():
    stored = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-01-02"]), "ticker": "AAPL", "close": [100.0, 100.0], "adj_close": [100.0, 100.0]})
    splits = pd.DataFrame({"ticker": ["AAPL"], "execution_date": pd.to_datetime(["2024-01-03"]), "split_from": [1], "split_to": [4]})
    out = mx.readjust_history(stored, splits, since="2024-01-02")
    assert out["adj_close"].tolist() == [25.0, 100.0]  # only rows before `since` are re-based here


def test_fundamentals_keyed_on_filing_date(client):
    f = mx.fetch_fundamentals(client, ["AAPL"])
    assert set(f["field"]) == {"revenue", "eps", "total_equity"}
    assert (f["available_date"] == pd.Timestamp("2024-02-02")).all() and (f["period_end"] == pd.Timestamp("2023-12-30")).all()


def test_membership_from_prices_and_snapshot_files(tmp_path):
    px = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]), "ticker": ["AAPL", "AAPL", "DEAD"]})
    meta = pd.DataFrame({"ticker": ["DEAD"], "delisted_date": pd.to_datetime(["2024-01-02"])})
    mem = mx.membership_from_prices(px, meta, "u").set_index("ticker")
    assert pd.isna(mem.loc["AAPL", "end_date"]) and mem.loc["DEAD", "end_date"] == pd.Timestamp("2024-01-02")

    f = tmp_path / "sp.csv"
    f.write_text("date,ticker\n2024-01-01,A\n2024-01-01,B\n2024-04-01,A\n2024-04-01,C\n")
    mem = load_membership(f, "idx").set_index("ticker")
    assert pd.isna(mem.loc["A", "end_date"]) and mem.loc["B", "end_date"] == pd.Timestamp("2024-03-31")
    assert mem.loc["C", "start_date"] == pd.Timestamp("2024-04-01")
    f2 = tmp_path / "flat.csv"
    f2.write_text("ticker\nA\nB\n")
    with pytest.warns(UserWarning, match="survivorship"):
        assert len(load_membership(f2, "x")) == 2


def test_api_error_status_is_raised():
    c = mx.MassiveClient(transport=lambda url: {"status": "NOT_AUTHORIZED", "message": "not entitled"})
    with pytest.raises(mx.MassiveError, match="not entitled"):
        c.get("/v2/aggs/grouped/locale/us/market/stocks/2024-01-02")
