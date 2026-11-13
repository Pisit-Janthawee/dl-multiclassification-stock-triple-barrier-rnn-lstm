"""Stationary, causal features (each value at day t uses data up to and including day t only)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------- Fractional differentiation
def ffd_weights(d: float, threshold: float = 1e-4, max_len: int = 500) -> np.ndarray:
    """Weights of the fixed-width window fractional difference (1 - B)^d.
    w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k ; stop when |w_k| < threshold."""
    w = [1.0]
    for k in range(1, max_len):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)


def frac_diff(series: pd.Series, d: float, threshold: float = 1e-3) -> pd.Series:
    """x~_t = sum_k w_k x_{t-k}  (NaN for the first len(w)-1 values)."""
    w = ffd_weights(d, threshold)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= len(w):
        out[len(w) - 1:] = np.convolve(x, w, mode="valid")
    return pd.Series(out, index=series.index)


def min_d_stationary(log_price: pd.Series, d_grid=np.arange(0, 1.01, 0.05), alpha=0.05, threshold=1e-3):
    """Smallest d whose fractionally differenced series passes the ADF test (keeps the most memory).
    Returns (d, table of ADF p-value and correlation with the original series per d)."""
    rows = []
    for d in d_grid:
        fd = frac_diff(log_price, d, threshold).dropna()
        p = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        corr = np.corrcoef(log_price.loc[fd.index], fd)[0, 1]
        rows.append({"d": round(d, 2), "adf_p": p, "corr_with_price": corr})
    table = pd.DataFrame(rows)
    ok = table[table.adf_p < alpha]
    return (float(ok.d.iloc[0]) if len(ok) else 1.0), table


# ---------------------------------------------------------------- Technical indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder): 100 - 100 / (1 + avg_gain / avg_loss)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def ticker_features(g: pd.DataFrame, market: pd.DataFrame, d: float) -> pd.DataFrame:
    """Feature table for ONE ticker. `g` and `market` have columns open/high/low/close/volume, date index."""
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=g.index)
    for k in (1, 5, 10, 21):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["vol_21"] = r.rolling(21).std()
    f["vol_ratio_5_21"] = r.rolling(5).std() / f["vol_21"]
    f["rsi_14"] = rsi(c, 14) / 100
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (sma20 - 2 * std20)) / (4 * std20)
    f["bb_width"] = 4 * std20 / sma20
    f["dist_sma_50"] = c / c.rolling(50).mean() - 1
    f["dist_sma_200"] = c / c.rolling(200).mean() - 1
    f["dist_ema_21"] = c / c.ewm(span=21, adjust=False).mean() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14).mean() / c
    logv = np.log1p(v)
    f["volume_z"] = (logv - logv.rolling(21).mean()) / logv.rolling(21).std()
    f["ffd_close"] = frac_diff(np.log(c), d)
    # market context
    mc = market["close"].reindex(g.index)
    mr = np.log(mc).diff()
    f["mkt_ret_1"] = mr
    f["mkt_ret_5"] = np.log(mc / mc.shift(5))
    f["mkt_vol_21"] = mr.rolling(21).std()
    f["rel_strength_21"] = f["ret_21"] - np.log(mc / mc.shift(21))
    f["beta_63"] = r.rolling(63).cov(mr) / mr.rolling(63).var()
    return f.replace([np.inf, -np.inf], np.nan)


FEATURES = ["ret_1", "ret_5", "ret_10", "ret_21", "vol_21", "vol_ratio_5_21", "rsi_14", "macd_hist",
            "bb_pctb", "bb_width", "dist_sma_50", "dist_sma_200", "dist_ema_21", "atr_14", "volume_z",
            "ffd_close", "mkt_ret_1", "mkt_ret_5", "mkt_vol_21", "rel_strength_21", "beta_63"]
