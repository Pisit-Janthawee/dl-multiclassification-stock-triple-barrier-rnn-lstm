"""Splitting, sequence building, metrics and the backtest."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

LABELS = [-1, 0, 1]


# ---------------------------------------------------------------- Split with purging
def assign_split(df: pd.DataFrame, valid_start: str, test_start: str) -> pd.Series:
    """Split by event start date, then PURGE events whose label end t1 reaches into the next set
    (their outcome would overlap the next period's prices)."""
    vs, ts = pd.Timestamp(valid_start), pd.Timestamp(test_start)
    split = np.select([df["date"] < vs, df["date"] < ts], ["train", "valid"], "test")
    purge = ((split == "train") & (df["t1"] >= vs)) | ((split == "valid") & (df["t1"] >= ts))
    return pd.Series(np.where(purge, "purged", split), index=df.index)


class PurgedKFold:
    """K-fold over time-ordered events with purging and an embargo (Lopez de Prado, ch. 7).

    For each test fold [a, b]:
      - purge: drop training events whose interval [t0, t1] overlaps the test period
      - embargo: also drop training events starting within `embargo` days after the test period
    """

    def __init__(self, n_splits=5, embargo_days=5):
        self.n_splits, self.embargo = n_splits, pd.Timedelta(days=embargo_days)

    def split(self, t0: pd.Series, t1: pd.Series):
        dates = np.sort(t0.unique())
        for chunk in np.array_split(dates, self.n_splits):
            a, b = chunk[0], chunk[-1]
            test = (t0 >= a) & (t0 <= b)
            overlap = (t0 <= b) & (t1 >= a)
            embargo = (t0 > b) & (t0 <= b + self.embargo)
            train = ~test & ~overlap & ~embargo
            yield np.flatnonzero(train.to_numpy()), np.flatnonzero(test.to_numpy())


# ---------------------------------------------------------------- Sequences for RNN / LSTM / GRU
def make_sequences(df: pd.DataFrame, features: list, lookback: int):
    """For each event row, the window of the last `lookback` days of features of the same ticker
    (ending at the event day). Rows without a full window are dropped.
    Returns X (n, lookback, n_features) and the row index it belongs to."""
    X, idx = [], []
    for _, g in df.groupby("ticker", sort=False):
        values = g[features].to_numpy(dtype=np.float32)
        for j in range(lookback - 1, len(g)):
            X.append(values[j - lookback + 1: j + 1])
            idx.append(g.index[j])
    return np.stack(X), np.array(idx)


# ---------------------------------------------------------------- Metrics
def score(y_true, proba, name="model", weights=None) -> dict:
    """proba columns ordered as LABELS (-1, 0, 1)."""
    pred = np.array(LABELS)[proba.argmax(1)]
    return {
        "model": name,
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=LABELS, sample_weight=weights),
        "tp_precision": ((pred == 1) & (y_true == 1)).sum() / max((pred == 1).sum(), 1),
        "tp_recall": ((pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1),
    }


# ---------------------------------------------------------------- Backtest
def backtest_long(events: pd.DataFrame, signal: pd.Series, prices: pd.DataFrame, cost_bps: float = 10.0):
    """Long-only, one position per ticker at a time.

    Enter at the close of the event day when `signal` is True and the ticker is flat; exit at the close
    of t1 (barrier touch or vertical barrier). `cost_bps` is charged on entry and on exit.
    Returns (daily portfolio returns, trades). Portfolio = equal weight across tickers, idle cash earns 0.
    """
    trades, daily = [], {}
    close = prices.pivot(index="date", columns="ticker", values="close")
    rets = np.log(close).diff()
    cost = cost_bps / 1e4
    for ticker, g in events[signal.reindex(events.index).fillna(False).to_numpy()].groupby("ticker"):
        held = pd.Series(0.0, index=close.index)
        busy_until = pd.Timestamp.min
        for _, e in g.sort_values("date").iterrows():
            if e["date"] <= busy_until:
                continue
            span = (close.index > e["date"]) & (close.index <= e["t1"])
            held[span] = rets.loc[span, ticker]
            first, last = np.flatnonzero(span)[[0, -1]]
            held.iloc[first] -= cost
            held.iloc[last] -= cost
            trades.append({"ticker": ticker, "entry": e["date"], "exit": e["t1"], "label": e["label"],
                           "ret": held[span].sum()})
            busy_until = e["t1"]
        daily[ticker] = held
    daily = pd.DataFrame(daily).reindex(columns=events["ticker"].unique()).fillna(0.0)
    return daily, pd.DataFrame(trades)


def performance(log_returns: pd.Series, periods: int = 252) -> dict:
    """CAGR, annualised volatility, Sharpe (rf = 0) and maximum drawdown of a daily log-return series."""
    r = log_returns.dropna()
    equity = np.exp(r.cumsum())
    years = len(r) / periods
    simple = np.exp(r) - 1
    return {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "ann_vol": simple.std() * np.sqrt(periods),
        "sharpe": simple.mean() / simple.std() * np.sqrt(periods) if simple.std() > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }
