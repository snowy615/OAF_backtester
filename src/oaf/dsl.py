"""A small, safe expression language for signals.

Expressions are parsed with :mod:`ast` and evaluated against a :class:`Panel`
through a whitelist of operators; there is no ``eval``. Names resolve to either
a strategy parameter (scalar) or a data field (dates x tickers frame).

Every operator is strictly backward-looking: the value at date ``t`` depends only
on rows ``<= t``, so a signal can never see the future. Time-series operators
(``ts_*``) work down each ticker's column; cross-sectional operators (``rank``,
``zscore`` ...) work across tickers on one date and only see universe members.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from .panel import Panel

Value = Union[pd.DataFrame, float, bool, str]

MAX_EXPR_LEN = 2000
MAX_WINDOW = 2520


class DSLError(ValueError):
    """Raised for anything wrong with an expression (syntax, names, types)."""


@dataclass(frozen=True)
class Op:
    name: str
    signature: str
    doc: str
    fn: Callable[..., Value]


OPS: dict[str, Op] = {}


def op(signature: str, doc: str):
    name = signature.split("(")[0]

    def deco(fn):
        OPS[name] = Op(name, signature, doc, fn)
        return fn

    return deco


# -- helpers -------------------------------------------------------------
def _frame(x: Value, who: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise DSLError(f"{who} expects a data field/expression, got the scalar {x!r}")
    return x.astype(float) if x.dtypes.eq(bool).all() else x


def _win(n: Value, who: str, lo: int = 1) -> int:
    if isinstance(n, (pd.DataFrame, str)) or isinstance(n, bool):
        raise DSLError(f"{who}: window must be a number or a param name")
    if not math.isfinite(n):
        raise DSLError(f"{who}: window is not finite")
    k = int(round(n))
    if k < lo or k > MAX_WINDOW:
        raise DSLError(f"{who}: window {k} outside [{lo}, {MAX_WINDOW}] (negative windows would look ahead)")
    return k


def _num(x: Value, who: str) -> float:
    if isinstance(x, (pd.DataFrame, str)):
        raise DSLError(f"{who} expects a number or a param name")
    return float(x)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


# -- time-series operators (per ticker, backward-looking) ------------------
@op("ts_mean(x, n)", "rolling mean of x over the last n days")
def _ts_mean(ev, x, n):
    return _frame(x, "ts_mean").rolling(_win(n, "ts_mean")).mean()


@op("ts_std(x, n)", "rolling standard deviation over n days")
def _ts_std(ev, x, n):
    return _frame(x, "ts_std").rolling(_win(n, "ts_std", 2)).std()


@op("ts_sum(x, n)", "rolling sum over n days")
def _ts_sum(ev, x, n):
    return _frame(x, "ts_sum").rolling(_win(n, "ts_sum")).sum()


@op("ts_min(x, n)", "rolling minimum over n days")
def _ts_min(ev, x, n):
    return _frame(x, "ts_min").rolling(_win(n, "ts_min")).min()


@op("ts_max(x, n)", "rolling maximum over n days")
def _ts_max(ev, x, n):
    return _frame(x, "ts_max").rolling(_win(n, "ts_max")).max()


@op("ts_median(x, n)", "rolling median over n days")
def _ts_median(ev, x, n):
    return _frame(x, "ts_median").rolling(_win(n, "ts_median")).median()


@op("ts_rank(x, n)", "percentile rank (0-1] of today's x within its own last n days")
def _ts_rank(ev, x, n):
    return _frame(x, "ts_rank").rolling(_win(n, "ts_rank", 2)).rank(pct=True)


@op("ts_zscore(x, n)", "(x - rolling mean) / rolling std over n days")
def _ts_zscore(ev, x, n):
    x = _frame(x, "ts_zscore")
    r = x.rolling(_win(n, "ts_zscore", 2))
    return _clean((x - r.mean()) / r.std())


@op("ts_delay(x, n)", "x as it was n days ago (n >= 1)")
def _ts_delay(ev, x, n):
    return _frame(x, "ts_delay").shift(_win(n, "ts_delay"))


@op("ts_delta(x, n)", "x minus x n days ago")
def _ts_delta(ev, x, n):
    x = _frame(x, "ts_delta")
    return x - x.shift(_win(n, "ts_delta"))


@op("ts_return(x, n)", "x / x n days ago - 1 (e.g. ts_return(close, 20) is 20-day momentum)")
def _ts_return(ev, x, n):
    x = _frame(x, "ts_return")
    return _clean(x / x.shift(_win(n, "ts_return")) - 1.0)


@op("ts_corr(x, y, n)", "rolling correlation of x and y over n days")
def _ts_corr(ev, x, y, n):
    return _clean(_frame(x, "ts_corr").rolling(_win(n, "ts_corr", 3)).corr(_frame(y, "ts_corr")))


@op("ts_skew(x, n)", "rolling skewness over n days")
def _ts_skew(ev, x, n):
    return _frame(x, "ts_skew").rolling(_win(n, "ts_skew", 3)).skew()


@op("ts_decay_linear(x, n)", "linearly-decayed weighted mean over n days (today weighted most)")
def _ts_decay(ev, x, n):
    x = _frame(x, "ts_decay_linear")
    k = _win(n, "ts_decay_linear")
    if k > 512:
        raise DSLError("ts_decay_linear: window capped at 512")
    out = sum((k - i) * x.shift(i) for i in range(k))
    return out / (k * (k + 1) / 2)


@op("ema(x, n)", "exponential moving average with span n")
def _ema(ev, x, n):
    k = _win(n, "ema")
    return _frame(x, "ema").ewm(span=k, adjust=False, min_periods=k).mean()


@op("rsi(x, n)", "Wilder RSI of x over n days, 0-100")
def _rsi(ev, x, n):
    k = _win(n, "rsi", 2)
    d = _frame(x, "rsi").diff()
    up = d.clip(lower=0).ewm(alpha=1 / k, adjust=False, min_periods=k).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / k, adjust=False, min_periods=k).mean()
    return 100 - 100 / (1 + _clean(up / dn))


# -- cross-sectional operators (across universe members on one date) -------
@op("rank(x)", "cross-sectional percentile rank in (0, 1] across the universe")
def _rank(ev, x):
    return ev.masked(_frame(x, "rank")).rank(axis=1, pct=True)


@op("zscore(x)", "cross-sectional z-score across the universe")
def _zscore(ev, x):
    x = ev.masked(_frame(x, "zscore"))
    return _clean(x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0))


@op("demean(x)", "x minus its cross-sectional mean")
def _demean(ev, x):
    x = ev.masked(_frame(x, "demean"))
    return x.sub(x.mean(axis=1), axis=0)


@op("scale(x)", "rescale so absolute values sum to 1 across the universe")
def _scale(ev, x):
    x = ev.masked(_frame(x, "scale"))
    return _clean(x.div(x.abs().sum(axis=1), axis=0))


@op("winsorize(x, k)", "clip x to within k cross-sectional standard deviations of the mean")
def _winsorize(ev, x, k):
    x = ev.masked(_frame(x, "winsorize"))
    k = _num(k, "winsorize")
    mu, sd = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=mu - k * sd, upper=mu + k * sd, axis=0)


@op('group_neutralize(x, "group")', 'x minus the mean of its group (e.g. "sector") on each date')
def _group_neutralize(ev, x, group):
    x = ev.masked(_frame(x, "group_neutralize"))
    if not isinstance(group, str) or group not in ev.panel.groups:
        raise DSLError(f"group_neutralize: unknown group {group!r}; available: {sorted(ev.panel.groups)}")
    labels = ev.panel.groups[group].reindex(x.columns).fillna("__none__")
    out = x.copy()
    for _, cols in labels.groupby(labels).groups.items():
        out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
    return out


# -- element-wise ----------------------------------------------------------
def _elementwise(name: str, f: Callable[[Any], Any]):
    def fn(ev, x):
        if isinstance(x, pd.DataFrame):
            with np.errstate(all="ignore"):
                return _clean(f(_frame(x, name)))
        return float(f(_num(x, name)))

    return fn


OPS["abs"] = Op("abs", "abs(x)", "absolute value", _elementwise("abs", np.abs))
OPS["sign"] = Op("sign", "sign(x)", "-1, 0 or 1", _elementwise("sign", np.sign))
OPS["log"] = Op("log", "log(x)", "natural log (NaN for x <= 0)", _elementwise("log", np.log))
OPS["sqrt"] = Op("sqrt", "sqrt(x)", "square root (NaN for x < 0)", _elementwise("sqrt", np.sqrt))


def _binary(name: str, f):
    def fn(ev, a, b):
        if isinstance(a, str) or isinstance(b, str):
            raise DSLError(f"{name} does not take strings")
        if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame):
            return float(f(a, b))
        return ev.like(f(ev.values(a), ev.values(b)))

    return fn


OPS["max"] = Op("max", "max(a, b)", "element-wise maximum", _binary("max", np.fmax))
OPS["min"] = Op("min", "min(a, b)", "element-wise minimum", _binary("min", np.fmin))


@op("clip(x, lo, hi)", "limit x to [lo, hi]")
def _clip(ev, x, lo, hi):
    return _frame(x, "clip").clip(lower=_num(lo, "clip"), upper=_num(hi, "clip"))


@op("where(cond, a, b)", "a where cond is true, else b")
def _where(ev, cond, a, b):
    cond = _frame(cond, "where")
    return ev.like(np.where(cond.fillna(0).astype(bool).values, ev.values(a), ev.values(b)))


@op("is_nan(x)", "true where x is missing")
def _is_nan(ev, x):
    return _frame(x, "is_nan").isna()


@op("nan_to(x, v)", "replace missing values of x with the number v")
def _nan_to(ev, x, v):
    return _frame(x, "nan_to").fillna(_num(v, "nan_to"))


# -- evaluator ---------------------------------------------------------------
_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
}
_CMP = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise DSLError("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise DSLError(f"expression longer than {MAX_EXPR_LEN} characters")
    try:
        return ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DSLError(f"syntax error in {expr!r}: {e.msg}") from None


def names_in(expr: str) -> set[str]:
    """Data-field / param names an expression refers to (function names excluded)."""
    tree = parse(expr)
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in funcs}


class Evaluator:
    def __init__(self, panel: Optional[Panel], params: dict[str, float]):
        self.panel = panel
        self.params = params

    # helpers used by operators
    def masked(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.where(self.panel.mask)

    def like(self, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=self.panel.dates, columns=self.panel.tickers)

    def values(self, v: Value):
        if isinstance(v, str):
            raise DSLError("strings are only valid as group names")
        return _frame(v, "value").values if isinstance(v, pd.DataFrame) else float(v)

    def eval(self, expr: str) -> Value:
        return self._eval(parse(expr).body)

    def _eval(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)):
                return node.value
            raise DSLError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            if isinstance(v, str):
                raise DSLError("cannot apply an operator to a string")
            if isinstance(node.op, ast.USub):
                return -_frame(v, "-") if isinstance(v, pd.DataFrame) else -v
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~self._bool(v) if isinstance(v, pd.DataFrame) else (not v)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            a, b = self._eval(node.left), self._eval(node.right)
            if isinstance(a, str) or isinstance(b, str):
                raise DSLError("cannot do arithmetic on a string")
            if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                a, b = self._bool(a), self._bool(b)
            else:
                a = _frame(a, "arithmetic") if isinstance(a, pd.DataFrame) else a
                b = _frame(b, "arithmetic") if isinstance(b, pd.DataFrame) else b
            try:
                with np.errstate(all="ignore"):
                    out = _BIN[type(node.op)](a, b)
            except ZeroDivisionError:
                raise DSLError("division by zero") from None
            except OverflowError:
                raise DSLError("numeric overflow") from None
            return _clean(out) if isinstance(out, pd.DataFrame) and not out.dtypes.eq(bool).all() else out
        if isinstance(node, ast.BoolOp):
            vals = [self._bool(self._eval(v)) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            out = None
            for o, right_node in zip(node.ops, node.comparators):
                if type(o) not in _CMP:
                    raise DSLError(f"unsupported comparison {type(o).__name__}")
                right = self._eval(right_node)
                if isinstance(left, str) or isinstance(right, str):
                    raise DSLError("cannot compare strings")
                res = _CMP[type(o)](left, right)
                out = res if out is None else (out & res)
                left = right
            return out
        if isinstance(node, ast.IfExp):
            return OPS["where"].fn(self, self._eval(node.test), self._eval(node.body), self._eval(node.orelse))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in OPS:
                name = getattr(node.func, "id", ast.dump(node.func))
                raise DSLError(f"unknown function {name!r}; available: {', '.join(sorted(OPS))}")
            if node.keywords:
                raise DSLError(f"{node.func.id}: keyword arguments are not supported")
            spec = OPS[node.func.id]
            if len(node.args) != spec.fn.__code__.co_argcount - 1:
                raise DSLError(f"wrong number of arguments: expected {spec.signature}")
            return spec.fn(self, *[self._eval(a) for a in node.args])
        raise DSLError(f"unsupported syntax: {type(node).__name__}")

    def _name(self, name: str) -> Value:
        if name in self.params:
            return float(self.params[name])
        if self.panel is not None and name in self.panel.fields:
            return self.panel.fields[name]
        fields = self.panel.field_names() if self.panel is not None else []
        raise DSLError(f"unknown name {name!r}; params: {sorted(self.params)}; fields: {fields}")

    def _bool(self, v: Value):
        if isinstance(v, pd.DataFrame):
            return v if v.dtypes.eq(bool).all() else v.fillna(0).astype(bool)
        if isinstance(v, str):
            raise DSLError("a string is not a condition")
        return bool(v)


# -- public entry points -----------------------------------------------------
def evaluate(expr: str, panel: Panel, params: dict[str, float]) -> pd.DataFrame:
    """Evaluate to a float (dates x tickers) frame, NaN outside the universe."""
    out = Evaluator(panel, params).eval(expr)
    if not isinstance(out, pd.DataFrame):
        raise DSLError(f"{expr!r} evaluates to a constant, not a per-ticker signal")
    return _clean(out.astype(float)).where(panel.mask)


def evaluate_condition(expr: str, panel: Panel, params: dict[str, float]) -> pd.DataFrame:
    """Evaluate a boolean expression; False outside the universe and where undefined."""
    ev = Evaluator(panel, params)
    out = ev.eval(expr)
    if not isinstance(out, pd.DataFrame):
        out = ev.like(np.full(panel.mask.shape, bool(out)))
    return ev._bool(out) & panel.mask


def evaluate_scalar(expr: str, params: dict[str, float]) -> float:
    """Evaluate an expression that may only mention params (construction knobs)."""
    out = Evaluator(None, params).eval(expr)
    if isinstance(out, (pd.DataFrame, str)):
        raise DSLError(f"{expr!r} must evaluate to a number")
    return float(out)


def operator_docs() -> str:
    """Reference text for the operators, used in Claude's system prompt and the README."""
    return "\n".join(f"- {o.signature}: {o.doc}" for o in OPS.values())
