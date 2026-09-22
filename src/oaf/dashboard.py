"""HTML dashboard: one report per run, plus an index comparing every run.

Everything is a single self-contained file (inline CSS and SVG, no JavaScript or
CDN), so a report can be attached to a pitch, opened offline, or served from any
static host. Charts are drawn straight from the return series.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import drawdown_series
from .report import fmt

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#6b7280;--line:#e5e7eb;--accent:#1d4ed8;--bench:#9ca3af;
--pos:#15803d;--neg:#b91c1c;--pos-bg:#dcfce7;--neg-bg:#fee2e2;--radius:12px}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e6e7ea;--muted:#9aa0ab;--line:#262a33;
--accent:#60a5fa;--bench:#6b7280;--pos:#4ade80;--neg:#f87171;--pos-bg:#14532d;--neg-bg:#7f1d1d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:15px;font-weight:600;margin:0 0 12px;color:var(--muted);
text-transform:uppercase;letter-spacing:.06em}.sub{color:var(--muted);margin:0}
.tag{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--line);font-size:12px;margin-right:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px}
.kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.kpi .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.kpi .h{font-size:12px;color:var(--muted)}.pos{color:var(--pos)}.neg{color:var(--neg)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{background:var(--card);border:1px solid var(--line);
border-radius:var(--radius);padding:18px 20px;grid-column:span 12}.c6{grid-column:span 6}.c4{grid-column:span 4}.c8{grid-column:span 8}
@media(max-width:800px){.c6,.c4,.c8{grid-column:span 12}}
svg{width:100%;height:auto;display:block}.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-bottom:6px}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;text-align:right;
border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}.heat td{text-align:center;padding:5px 6px;border-radius:4px;border:2px solid var(--card)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--line);padding:2px 6px;border-radius:4px}
.note{color:var(--muted);font-size:13px}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{overflow-x:auto}dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;margin:0}dt{color:var(--muted)}dd{margin:0}
"""


# -- SVG primitives -----------------------------------------------------------------
def _fmt_axis(v: float, pct: bool) -> str:
    if pct:
        return f"{v:+.0%}" if abs(v) < 0.995 else f"{v:+.1%}"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


def _line_chart(series: dict[str, pd.Series], colors: Sequence[str], height: int = 260, pct: bool = False, fill: bool = False, zero: bool = False) -> str:
    """Multi-series line chart with a light grid. ``series`` share a DatetimeIndex."""
    W, H, L, R, T, B = 960, height, 56, 12, 12, 28
    frames = [s.dropna() for s in series.values() if s is not None and s.notna().any()]
    if not frames:
        return "<p class='note'>no data</p>"
    idx = frames[0].index
    lo = min(float(s.min()) for s in frames)
    hi = max(float(s.max()) for s in frames)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        hi = lo + 1e-9
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    x0, x1 = idx[0].value, idx[-1].value
    sx = lambda t: L + (t - x0) / max(x1 - x0, 1) * (W - L - R)  # noqa: E731
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)  # noqa: E731

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">']
    ticks = np.linspace(lo + pad, hi - pad, 5 if height >= 220 else 3)
    for tv in ticks:
        y = sy(tv)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="var(--line)" stroke-width="1"/>')
        parts.append(f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" fill="var(--muted)" text-anchor="end">{_fmt_axis(tv, pct)}</text>')
    if zero and lo < 0 < hi:
        parts.append(f'<line x1="{L}" y1="{sy(0):.1f}" x2="{W-R}" y2="{sy(0):.1f}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>')
    years = pd.date_range(idx[0], idx[-1], freq="YS")
    for d in years[:: max(1, len(years) // 10)]:
        x = sx(d.value)
        parts.append(f'<text x="{x:.1f}" y="{H-8}" font-size="11" fill="var(--muted)" text-anchor="middle">{d.year}</text>')
    for (name, s), color in zip(series.items(), colors):
        if s is None or not s.notna().any():
            continue
        s = s.dropna()
        step = max(1, len(s) // 1200)
        pts = [(sx(t.value), sy(v)) for t, v in zip(s.index[::step], s.values[::step])]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if fill:
            parts.append(f'<path d="{d} L{pts[-1][0]:.1f},{sy(0):.1f} L{pts[0][0]:.1f},{sy(0):.1f} Z" fill="{color}" opacity="0.15"/>')
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"><title>{html.escape(name)}</title></path>')
    parts.append("</svg>")
    return "".join(parts)


def _sparkline(s: pd.Series, w: int = 140, h: int = 34) -> str:
    s = s.dropna()
    if len(s) < 2:
        return ""
    lo, hi = float(s.min()), float(s.max())
    hi = hi if hi > lo else lo + 1e-9
    xs = np.linspace(0, w, len(s))
    ys = h - (s.values - lo) / (hi - lo) * (h - 4) - 2
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    color = "var(--pos)" if s.iloc[-1] >= s.iloc[0] else "var(--neg)"
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>'


def _legend(items: Sequence[tuple[str, str]]) -> str:
    return "<div class='legend'>" + "".join(f"<span><i style='background:{c}'></i>{html.escape(n)}</span>" for n, c in items) + "</div>"


# -- tables ---------------------------------------------------------------------------
def _monthly_heatmap(returns: pd.Series) -> str:
    m = (1 + returns).resample("ME").prod() - 1
    if m.empty:
        return ""
    table = m.groupby([m.index.year, m.index.month]).first().unstack()
    table = table.reindex(columns=range(1, 13))
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    scale = max(float(np.nanmax(np.abs(table.values))), 1e-9)
    head = "".join(f"<th>{pd.Timestamp(2000, mth, 1):%b}</th>" for mth in range(1, 13)) + "<th>Year</th>"
    rows = []
    for year, row in table.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("<td></td>")
            else:
                a = min(abs(v) / scale, 1.0) * 0.85 + 0.1
                bg = f"color-mix(in srgb, var(--{'pos' if v >= 0 else 'neg'}-bg) {a*100:.0f}%, var(--card))"
                cells.append(f"<td style='background:{bg}'>{v:+.1%}</td>")
        y = yearly.get(year, np.nan)
        cells.append(f"<td class='{'pos' if y >= 0 else 'neg'}'><b>{y:+.1%}</b></td>")
        rows.append(f"<tr><td><b>{year}</b></td>{''.join(cells)}</tr>")
    return f"<div class='wrap'><table class='heat'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _kpi(label: str, key: str, metrics: dict, hint: str = "", signed: bool = True) -> str:
    v = metrics.get(key)
    cls = ""
    if signed and isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
    if key == "max_drawdown":
        cls = "neg"
    return f"<div class='kpi'><div class='l'>{label}</div><div class='v {cls}'>{fmt(key, v)}</div><div class='h'>{hint}</div></div>"


def _params_table(spec) -> str:
    rows = "".join(
        f"<tr><td><code>{p.name}</code></td><td>{p.value:g}</td><td>{p.low:g} – {p.high:g}</td><td style='text-align:left;white-space:normal'>{html.escape(p.description)}</td></tr>"
        for p in spec.params
    )
    return f"<div class='wrap'><table><thead><tr><th>Param</th><th>Value</th><th>Sweep range</th><th style='text-align:left'>Meaning</th></tr></thead><tbody>{rows}</tbody></table></div>"


_LABELS = {
    "sharpe": "Sharpe", "deflated_sharpe": "Deflated", "max_drawdown": "Max DD", "annualised_return": "Ann. return",
    "ic_mean": "IC", "information_ratio": "IR", "annual_turnover": "Turnover", "is_score": "In-sample", "oos_score": "Out-of-sample",
    "train_start": "Train from", "test_start": "Test from", "test_end": "Test to", "trial": "Trial", "cumulative_return": "Cum. return",
}


def _df_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    """Metric columns get their report formatting; knob columns print as plain numbers."""

    def cell(col, v):
        if isinstance(v, float) and math.isnan(v):
            return ""
        if col in _METRIC_COLS or col in ("is_score", "oos_score"):
            return fmt(col if col in _METRIC_COLS else "sharpe", v)
        if isinstance(v, float):
            return f"{v:g}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(_LABELS.get(str(c), str(c).replace('sim.', '').replace('_', ' ')))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(c, v)}</td>" for c, v in zip(df.columns, row)) + "</tr>"
        for row in df.head(max_rows).itertuples(index=False)
    )
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _page(title: str, body: str, subtitle: str = "") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}
<p class='note' style='margin-top:32px'>Oxford Alpha Fund backtester · generated {pd.Timestamp.now():%Y-%m-%d %H:%M}</p></main></body></html>"""


# -- run report -------------------------------------------------------------------------
def run_report(result, sweep_table: Optional[pd.DataFrame] = None, walk_forward=None, title: Optional[str] = None) -> str:
    spec, cfg, m = result.spec, result.config, result.metrics
    r = result.returns.loc[m.get("start", result.returns.index[0]) :]
    bench = result.benchmark.reindex(r.index).fillna(0.0)
    growth = (1 + r).cumprod()
    bench_growth = (1 + bench).cumprod()
    dd = drawdown_series(r)
    roll = r.rolling(126).mean() / r.rolling(126).std() * math.sqrt(cfg.periods_per_year)
    gross = result.weights.abs().sum(axis=1).reindex(r.index) * result.exposure_scale.reindex(r.index)

    overlay = []
    if cfg.vol_target is not None:
        overlay.append(f"vol target {cfg.vol_target:.0%}")
    if cfg.dd_limit is not None:
        overlay.append(f"drawdown breaker −{cfg.dd_limit:.0%}")
    tags = [spec.universe.name, spec.rebalance, f"{spec.construction.mode} · {spec.construction.direction.replace('_', '/')}", *overlay]
    if spec.universe.min_adv:
        tags.append(f"ADV ≥ ${spec.universe.min_adv/1e6:.0f}m")

    kpis = "".join(
        [
            _kpi("Sharpe", "sharpe", m, f"deflated {fmt('deflated_sharpe', m.get('deflated_sharpe'))}" if m.get("n_trials", 1) > 1 else f"PSR {fmt('psr', m.get('psr'))}"),
            _kpi("Annualised return", "annualised_return", m, f"vs {fmt('benchmark_annualised_return', m.get('benchmark_annualised_return'))} equal-weight"),
            _kpi("Max drawdown", "max_drawdown", m, f"Calmar {fmt('calmar', m.get('calmar'))}"),
            _kpi("Cumulative return", "cumulative_return", m, f"{m.get('start','')} → {m.get('end','')}"),
            _kpi("Information coefficient", "ic_mean", m, f"t-stat {fmt('ic_tstat', m.get('ic_tstat'))} · hit {fmt('ic_hit_rate', m.get('ic_hit_rate'))}"),
            _kpi("Information ratio", "information_ratio", m, f"vol {fmt('annualised_vol', m.get('annualised_vol'))} · turnover {fmt('annual_turnover', m.get('annual_turnover'))}×"),
        ]
    )
    rules = ""
    if spec.rules:
        rules = "".join(f"<dt>{k.replace('_', ' ')}</dt><dd><code>{html.escape(v)}</code></dd>" for k, v in spec.rules.model_dump().items() if v)

    body = f"""
<header><div><h1>{html.escape(title or spec.name)}</h1><p class='sub'>{html.escape(spec.description or spec.idea)}</p></div>
<div>{''.join(f"<span class='tag'>{html.escape(t)}</span>" for t in tags)}</div></header>
<div class='kpis'>{kpis}</div>
<div class='grid'>
<div class='card c8'><h2>Growth of 1</h2>{_legend([(spec.name, 'var(--accent)'), ('equal-weight universe', 'var(--bench)')])}
{_line_chart({spec.name: growth, 'benchmark': bench_growth}, ['var(--accent)', 'var(--bench)'])}</div>
<div class='card c4'><h2>Signal &amp; rules</h2><dl><dt>signal</dt><dd><code>{html.escape(spec.signal)}</code></dd>{rules}
<dt>hypothesis</dt><dd class='note'>{html.escape(spec.hypothesis or '—')}</dd></dl></div>
<div class='card c6'><h2>Drawdown</h2>{_line_chart({'drawdown': dd}, ['var(--neg)'], height=180, pct=True, fill=True, zero=True)}</div>
<div class='card c6'><h2>Rolling 6-month Sharpe</h2>{_line_chart({'sharpe': roll}, ['var(--accent)'], height=180, zero=True)}</div>
<div class='card'><h2>Monthly returns</h2>{_monthly_heatmap(r)}</div>
<div class='card c6'><h2>Gross exposure (after risk overlay)</h2>{_line_chart({'gross': gross}, ['var(--accent)'], height=160, zero=True)}</div>
<div class='card c6'><h2>Parameters</h2>{_params_table(spec)}
<p class='note'>Costs {cfg.commission_bps + cfg.slippage_bps:g} bps/turnover · borrow {cfg.borrow_bps_annual:g} bps/yr · lag {cfg.execution_lag}d ·
max weight {spec.construction.max_weight} · gross {spec.construction.gross_exposure}</p></div>
"""
    if sweep_table is not None and len(sweep_table):
        knobs = [c for c in sweep_table.columns if c not in _METRIC_COLS]
        cols = knobs + [c for c in ("sharpe", "deflated_sharpe", "max_drawdown", "annualised_return", "ic_mean", "information_ratio", "annual_turnover") if c in sweep_table]
        body += f"<div class='card'><h2>Sweep · {len(sweep_table)} configurations (in sample, best first)</h2>{_df_table(sweep_table[cols])}<p class='note'>Chosen in sample — read the deflated Sharpe, and confirm out of sample with <code>oaf walkforward</code>.</p></div>"
    if walk_forward is not None:
        wf = walk_forward
        oos = (1 + wf.oos_returns).cumprod()
        body += f"""<div class='card'><p class='note'>The sections above are the full-sample run of the spec's default parameters. Below, each fold re-picks the best configuration on its training window and is scored only on the following unseen window.</p></div>
<div class='card c8'><h2>Walk-forward · stitched out-of-sample growth</h2>{_line_chart({'oos': oos}, ['var(--accent)'])}</div>
<div class='card c4'><h2>Out of sample</h2><div class='kpis' style='grid-template-columns:1fr 1fr'>{_kpi('Sharpe', 'sharpe', wf.metrics)}{_kpi('Max DD', 'max_drawdown', wf.metrics)}
{_kpi('Ann. return', 'annualised_return', wf.metrics)}{_kpi('IR', 'information_ratio', wf.metrics)}</div>
<p class='note'>{len(wf.folds)} folds · mean OOS − IS {wf.objective}: <b class='{'pos' if wf.degradation >= 0 else 'neg'}'>{wf.degradation:+.2f}</b></p></div>
<div class='card'><h2>Folds</h2>{_df_table(wf.folds)}</div>"""
    if spec.assumptions:
        body += "<div class='card'><h2>Assumptions</h2><ul class='note'>" + "".join(f"<li>{html.escape(a)}</li>" for a in spec.assumptions) + "</ul></div>"
    body += "</div>"
    return _page(title or spec.name, body)


_METRIC_COLS = {
    "sharpe", "max_drawdown", "cumulative_return", "annualised_return", "annualised_vol", "sortino", "calmar", "hit_rate",
    "information_ratio", "benchmark_annualised_return", "ic_mean", "ic_std", "ic_ir", "ic_tstat", "ic_hit_rate", "annual_turnover",
    "psr", "n_trials", "deflated_sharpe", "start", "end",
}


# -- index across runs ---------------------------------------------------------------
def build_index(runs_dir: str | Path) -> Path:
    runs_dir = Path(runs_dir)
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        mfile = d / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        eq = None
        if (d / "oos_returns.csv").exists():  # walk-forward: show the stitched out-of-sample path
            eq = (1 + pd.read_csv(d / "oos_returns.csv", index_col="date", parse_dates=True)["oos_return"]).cumprod()
        elif (d / "returns.csv").exists():
            eq = pd.read_csv(d / "returns.csv", index_col="date", parse_dates=True)["equity"]
        kind = "walk-forward" if (d / "folds.csv").exists() else ("sweep" if (d / "sweep.csv").exists() else "backtest")
        rows.append((d.name, kind, m, eq, (d / "report.html").exists()))
    rows.sort(key=lambda r: -(r[2].get("sharpe") or -9))
    if not rows:
        body = "<header><h1>Strategy runs</h1></header><p class='note'>No runs yet - run <code>oaf backtest</code>.</p>"
    else:
        trs = []
        for name, kind, m, eq, has_report in rows:
            link = f"<a href='{html.escape(name)}/report.html'>{html.escape(name)}</a>" if has_report else html.escape(name)
            cells = [
                f"<td>{link}<br><span class='note'>{kind} · {m.get('start','')} → {m.get('end','')}</span></td>",
                f"<td>{_sparkline(eq) if eq is not None else ''}</td>",
            ]
            for key in ("sharpe", "deflated_sharpe", "annualised_return", "max_drawdown", "ic_mean", "information_ratio", "annual_turnover"):
                v = m.get(key)
                cls = "neg" if key == "max_drawdown" else ("pos" if isinstance(v, (int, float)) and v > 0 else ("neg" if isinstance(v, (int, float)) and v < 0 else ""))
                cells.append(f"<td class='{cls}'>{fmt(key, v).replace('n/a', '—')}</td>")
            trs.append("<tr>" + "".join(cells) + "</tr>")
        body = f"""<header><div><h1>Strategy runs</h1><p class='sub'>{len(rows)} runs in <code>{html.escape(str(runs_dir))}</code>, best Sharpe first</p></div></header>
<div class='card'><div class='wrap'><table><thead><tr><th>Run</th><th>Equity</th><th>Sharpe</th><th>Deflated</th><th>Ann. return</th><th>Max DD</th><th>IC</th><th>IR</th><th>Turnover</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div></div>"""
    out = runs_dir / "index.html"
    out.write_text(_page("Strategy runs", body))
    return out


def open_in_browser(path: str | Path) -> None:
    import webbrowser

    webbrowser.open(Path(path).resolve().as_uri())
