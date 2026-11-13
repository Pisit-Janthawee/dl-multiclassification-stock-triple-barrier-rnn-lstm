# Stock Triple-Barrier Classification (RNN / LSTM / GRU)

> Work in progress: notebooks 01-03 are complete; 04 (modeling) and 05 (evaluation and backtest) are being built.

Second version of my stock forecasting project. Instead of predicting the scaled closing price (the first version
reached R² ≈ 0.99, which a "tomorrow = today" guess also achieves on price levels), each trading day is labelled with
the **triple-barrier method**: would a trade started that day hit its take-profit, its stop-loss, or run out of time first?

## Data

Daily prices from Yahoo Finance via `yfinance`, 2010-01-01 to 2026-11-11, for 14 stocks
(NVDA, CRWD, AMD, ADBE, MSFT, GOOGL, INTC, AAPL, AMZN, CRM, IBM, TSLA, PYPL, XYZ) plus SPY.
The data is not committed; notebook 01 downloads it into `data/raw/`.

## Notebooks

| Notebook | Content |
|---|---|
| [01 Data Understanding](notebooks/01_data_understanding.ipynb) | Data health, fat tails, stationarity, volatility clustering, why R² on prices misleads |
| [02 Triple-Barrier Labeling](notebooks/02_triple_barrier_labeling.ipynb) | Volatility-scaled barriers, brute-force check, sensitivity, overlap and uniqueness weights, CUSUM filter |
| [03 Features & Purged Split](notebooks/03_features_and_split.ipynb) | 21 stationary features, fractional differentiation, Kruskal-Wallis effects, purged time split |
| 04 Modeling | in progress |
| 05 Evaluation & Backtest | in progress |

## How to run

Python 3.10. Keras runs on the PyTorch backend (GPU if available).

```bash
pip install -r requirements.txt
# run the notebooks in order; notebook 01 downloads the data
```
