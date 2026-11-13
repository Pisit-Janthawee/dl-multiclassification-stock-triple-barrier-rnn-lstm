"""Download daily prices with yfinance and cache them as Parquet (data/raw is git-ignored)."""
import pandas as pd

from .config import END, MARKET, PROCESSED_DIR, RAW_DIR, START, TICKERS

RAW_FILE = RAW_DIR / "prices.parquet"


def download_prices(tickers=None, start=START, end=END, refresh=False) -> pd.DataFrame:
    """Long table: date, ticker, open, high, low, close, volume (split/dividend adjusted).

    Cached in data/raw/prices.parquet; set refresh=True to download again.
    """
    if RAW_FILE.exists() and not refresh:
        return pd.read_parquet(RAW_FILE)
    import yfinance as yf

    tickers = tickers or TICKERS + [MARKET]
    wide = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    long = (wide.stack(level="Ticker", future_stack=True)
                .rename_axis(["date", "ticker"]).reset_index())
    long.columns = [c.lower() for c in long.columns]
    long = long.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    long = long[["date", "ticker", "open", "high", "low", "close", "volume"]]
    long.to_parquet(RAW_FILE, index=False)
    return long


def save_processed(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
