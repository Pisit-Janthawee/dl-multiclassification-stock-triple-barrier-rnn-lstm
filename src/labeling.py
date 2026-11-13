"""Triple-barrier labeling, sample uniqueness and the CUSUM filter (Lopez de Prado, *Advances in Financial ML*, ch. 2-4).

For an event starting at day t0 with close P0 and daily volatility s:
    upper barrier  = P0 * exp(+pt_mult * s * sqrt(horizon))   -> label +1 (take-profit)
    lower barrier  = P0 * exp(-sl_mult * s * sqrt(horizon))   -> label -1 (stop-loss)
    vertical       = t0 + horizon trading days                -> label  0 (time-out)
The label is the barrier touched FIRST by the closing price on days t0+1 ... t0+horizon.
"""
import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """EWM standard deviation of daily log returns (uses data up to and including t)."""
    return np.log(close).diff().ewm(span=span, min_periods=span).std()


def triple_barrier(close: pd.Series, vol: pd.Series, horizon: int = 10,
                   pt_mult: float = 2.0, sl_mult: float = 2.0) -> pd.DataFrame:
    """Label every day of ONE ticker. Returns a DataFrame indexed like `close` with:
    label (-1/0/1), t1 (date the label is known = touch date or vertical barrier),
    days_held, ret (log return from t0 to t1), barrier (width in log-return units)."""
    logp = np.log(close.to_numpy())
    v = vol.to_numpy()
    dates = close.index
    n = len(close)
    label = np.full(n, np.nan)
    t1_idx = np.full(n, -1)
    ret = np.full(n, np.nan)
    width = v * np.sqrt(horizon)
    for i in range(n - horizon):            # last `horizon` days have no complete future -> unlabeled
        if np.isnan(width[i]):
            continue
        path = logp[i + 1: i + 1 + horizon] - logp[i]
        up = np.flatnonzero(path >= pt_mult * width[i])
        dn = np.flatnonzero(path <= -sl_mult * width[i])
        first_up = up[0] if len(up) else horizon
        first_dn = dn[0] if len(dn) else horizon
        if first_up < first_dn:
            label[i], k = 1, first_up
        elif first_dn < first_up:
            label[i], k = -1, first_dn
        else:                                 # neither barrier touched (equal only when both == horizon)
            label[i], k = 0, horizon - 1
        t1_idx[i] = i + 1 + k
        ret[i] = path[k]
    out = pd.DataFrame({"label": label, "ret": ret, "barrier": width}, index=dates)
    valid = t1_idx >= 0
    out["t1"] = pd.NaT
    out.loc[valid, "t1"] = dates[t1_idx[valid]]
    out["days_held"] = np.where(valid, t1_idx - np.arange(n), np.nan)
    return out


def average_uniqueness(t0: pd.Index, t1: pd.Series, all_dates: pd.Index) -> pd.Series:
    """Average uniqueness of each event (one ticker).

    Concurrency c_t = number of events whose holding interval (t0, t1] contains day t.
    Uniqueness of event i = mean over its interval of 1 / c_t  (1 = no overlap, 1/H = fully overlapped).
    """
    pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
    start = pos.reindex(t0).to_numpy() + 1          # holding starts the day after t0
    end = pos.reindex(t1).to_numpy()
    conc = np.zeros(len(all_dates) + 1)
    np.add.at(conc, start, 1)
    np.add.at(conc, end + 1, -1)
    conc = np.cumsum(conc)[:-1]
    inv = np.where(conc > 0, 1 / np.maximum(conc, 1), 0)
    csum = np.concatenate([[0], np.cumsum(inv)])
    u = (csum[end + 1] - csum[start]) / (end - start + 1)
    return pd.Series(u, index=t0)


def cusum_filter(close: pd.Series, threshold: pd.Series | float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter on log returns: emit an event when the cumulative up- or down-move
    since the last event exceeds `threshold` (a number or a per-day Series)."""
    r = np.log(close).diff().dropna()
    h = threshold.reindex(r.index) if isinstance(threshold, pd.Series) else pd.Series(threshold, index=r.index)
    events, s_pos, s_neg = [], 0.0, 0.0
    for t, x in r.items():
        if np.isnan(h[t]):
            continue
        s_pos, s_neg = max(0.0, s_pos + x), min(0.0, s_neg + x)
        if s_pos > h[t]:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -h[t]:
            s_neg = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def label_all(prices: pd.DataFrame, vol_span=20, horizon=10, pt_mult=2.0, sl_mult=2.0) -> pd.DataFrame:
    """Triple-barrier labels + uniqueness for every ticker in a long price table."""
    parts = []
    for ticker, g in prices.groupby("ticker", sort=True):
        close = g.set_index("date")["close"]
        vol = daily_volatility(close, vol_span)
        lab = triple_barrier(close, vol, horizon, pt_mult, sl_mult)
        lab["vol"] = vol
        lab = lab.dropna(subset=["label"])
        lab["uniqueness"] = average_uniqueness(lab.index, lab["t1"], close.index).to_numpy()
        lab["ticker"] = ticker
        parts.append(lab.rename_axis("date").reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["label"] = out["label"].astype(int)
    return out
