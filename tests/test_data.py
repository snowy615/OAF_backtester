import io

import numpy as np
import pandas as pd
import pytest

from oaf.data.ingest import ColumnMapping, apply_mapping, clean_prices, infer_mapping_heuristic
from oaf.data.warehouse import ASSUMED_DELISTING_RETURN, Warehouse

MESSY = """Trade Date,Symbol,Open Price,High,Low,PX_LAST,Vol
03/01/2024,abc ,10,10.5,9.8,10.2,"1,000"
04/01/2024,abc ,10.2,10.1,10.6,10.4,1200
04/01/2024,abc ,10.2,10.1,10.6,10.5,1200
05/01/2024,abc ,10.5,55,10.4,50.0,900
08/01/2024,abc ,10.5,10.9,10.4,10.6,950
06/01/2024,abc ,10.5,10.9,10.4,10.6,950
09/01/2024,abc ,10.6,10.9,10.4,-1,950
not a date,abc ,1,1,1,1,1
"""


def test_heuristic_mapping_and_cleaning_of_a_messy_file():
    raw = pd.read_csv(io.StringIO(MESSY))
    mapping = infer_mapping_heuristic(raw)
    assert (mapping.date_col, mapping.ticker_col, mapping.close_col, mapping.volume_col) == ("Trade Date", "Symbol", "PX_LAST", "Vol")
    mapping.dayfirst = True
    clean, report = clean_prices(apply_mapping(raw, mapping))

    assert clean["ticker"].unique().tolist() == ["ABC"]
    assert clean["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04", "2024-01-08"]
    assert clean["close"].tolist() == [10.2, 10.5, 10.6]  # duplicate keeps the later row
    assert clean["volume"].iloc[0] == 1000
    assert clean.iloc[1][["high", "low"]].tolist() == [10.6, 10.1]  # swapped back
    assert set(report.dropped) == {
        "unparseable date/ticker", "missing/non-positive close", "weekend date", "duplicate (date, ticker)",
        "bad tick (spike that reverses next day)",
    }


def test_wide_layout_and_pence_scaling():
    raw = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "VOD": [7000, 7100], "BP": [45000, 46000]})
    mapping = infer_mapping_heuristic(raw)
    assert mapping.layout == "wide"
    mapping.price_scale = 0.01
    out = apply_mapping(raw, mapping).sort_values(["ticker", "date"])
    assert out["adj_close"].tolist() == [450.0, 460.0, 70.0, 71.0]
    assert out["close"].tolist() == out["adj_close"].tolist()


def test_unadjusted_split_is_flagged():
    dates = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"date": dates, "ticker": "X", "open": np.nan, "high": np.nan, "low": np.nan,
                       "close": [100, 101, 50.4, 50.0, 51, 52.0], "adj_close": np.nan, "volume": 1.0})
    _, report = clean_prices(df)
    assert report.flagged == {"possible unadjusted split (no adj_close)": 1}


def _prices(ticker, dates, closes, adj=None):
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": closes, "high": closes, "low": closes,
                         "close": closes, "adj_close": closes if adj is None else adj, "volume": 1000.0})


@pytest.fixture
def small_wh(tmp_path):
    wh = Warehouse(tmp_path / "wh")
    d = pd.bdate_range("2024-01-01", periods=10)
    wh.write_prices(pd.concat([
        _prices("LIVE", d, np.linspace(100, 109, 10)),
        _prices("DEAD", d[:5], [50, 49, 48, 47, 46.0]),
        _prices("SPLIT", d, [100, 102, 51.5, 52, 52, 52, 52, 52, 52, 52.0], adj=[50, 51, 51.5, 52, 52, 52, 52, 52, 52, 52.0]),
    ]))
    wh.write_meta(pd.DataFrame({"ticker": ["LIVE", "DEAD", "SPLIT"], "name": "", "sector": ["a", "a", "b"],
                                "delisted_date": [pd.NaT, d[4], pd.NaT], "delisting_return": [np.nan, np.nan, np.nan]}))
    wh.write_membership(pd.DataFrame({"universe": "idx", "ticker": ["LIVE", "DEAD"],
                                      "start_date": [d[3], d[0]], "end_date": [pd.NaT, d[2]]}))
    wh.write_fundamentals(pd.DataFrame({"ticker": "LIVE", "field": "eps", "period_end": [d[0], d[0]],
                                        "available_date": [d[4], d[7]], "value": [1.0, 1.5]}))
    return wh, d


def test_fundamentals_are_point_in_time(small_wh):
    wh, d = small_wh
    eps = wh.load_panel().fields["eps"]["LIVE"]
    assert eps.loc[:d[3]].isna().all()  # period ended on d0 but nobody knew until d4
    assert eps.loc[d[4]:d[6]].eq(1.0).all()
    assert eps.loc[d[7]:].eq(1.5).all()  # the restatement only applies from its own publication date


def test_membership_mask_is_point_in_time(small_wh):
    wh, d = small_wh
    mask = wh.load_panel("idx").mask
    assert mask["LIVE"].tolist() == [False] * 3 + [True] * 7
    assert mask["DEAD"].tolist() == [True] * 3 + [False] * 7
    assert not mask["SPLIT"].any()


def test_delisting_and_split_handling(small_wh):
    wh, d = small_wh
    panel = wh.load_panel()
    assert panel.returns.loc[d[5], "DEAD"] == ASSUMED_DELISTING_RETURN
    assert panel.returns["DEAD"].loc[d[6]:].isna().all()
    assert panel.returns.loc[d[2], "SPLIT"] == pytest.approx(51.5 / 51 - 1)  # not -50%
    assert panel.fields["raw_close"].loc[d[1], "SPLIT"] == 102


def test_upsert_and_survivorship_warning(tmp_path):
    wh = Warehouse(tmp_path / "wh")
    d = pd.bdate_range("2019-01-01", "2024-01-01")
    wh.write_prices(_prices("A", d, np.full(len(d), 10.0)))
    assert wh.write_prices(_prices("A", d[-3:], [11, 11, 11.0])) == len(d)
    assert wh.read("prices")["close"].iloc[-1] == 11
    report = wh.survivorship_report()
    assert len(report.warnings) == 2 and "survivors-only" in report.warnings[0]


def test_demo_warehouse_is_healthy(warehouse):
    report = warehouse.survivorship_report()
    assert report.n_delisted > 0 and report.has_membership and not report.warnings
