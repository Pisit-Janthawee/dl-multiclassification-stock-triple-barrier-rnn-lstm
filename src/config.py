"""Project paths and constants shared by every notebook."""
from pathlib import Path

RANDOM_STATE = 42

# ---------------------------------------------------------------- Universe & period
TICKERS = ["NVDA", "CRWD", "AMD", "ADBE", "MSFT", "GOOGL",            # IT stocks of my first version
           "INTC", "AAPL", "AMZN", "CRM", "IBM", "TSLA", "PYPL", "XYZ"]  # XYZ = Block (ticker SQ until 2025)
MARKET = "SPY"                     # market context (S&P 500 ETF)
START = "2010-01-01"
END = "2026-11-12"                 # fixed end date -> reproducible downloads (exclusive in yfinance)

# ---------------------------------------------------------------- Triple-barrier settings
VOL_SPAN = 20                      # EWM span (days) of the daily volatility estimate
PT_MULT, SL_MULT = 0.75, 0.75      # barrier = mult x daily vol x sqrt(horizon); 0.75 gives balanced classes
HORIZON = 10                       # vertical barrier (trading days)
CLASSES = {-1: "stop-loss", 0: "time-out", 1: "take-profit"}

# ---------------------------------------------------------------- Time split (by event start date)
VALID_START = "2022-01-01"
TEST_START = "2024-01-01"

# ---------------------------------------------------------------- Paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Colours
LABEL_COLORS = {-1: "#dc2626", 0: "#9ca3af", 1: "#15803d"}   # stop-loss red, time-out grey, take-profit green
PRICE_COLOR = "#374151"
