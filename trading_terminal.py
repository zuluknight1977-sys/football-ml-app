#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      CROSS-ASSET TRADING TERMINAL  —  macOS Native Edition                 ║
║      Single-File Production Script  |  Virtual Balance: $10,000            ║
║      Supports: Apple Silicon (M1/M2/M3/M4) + Intel  |  PyQt6 Dark UI      ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  macOS SETUP & COMPILATION INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Open Terminal.app and run the following commands in order:

  ── Step 1: Create & activate an isolated virtual environment ──────────────
  python3 -m venv trading_env
  source trading_env/bin/activate

  ── Step 2: Upgrade pip and install all dependencies ──────────────────────
  pip install --upgrade pip setuptools wheel
  pip install PyQt6 \
              pandas \
              numpy \
              yfinance \
              pandas_datareader \
              pandas_ta \
              matplotlib \
              pyinstaller

  ── Step 3: Launch the application directly ───────────────────────────────
  python trading_terminal.py

  ── Step 4: Bundle into a native standalone .app (Universal Binary) ───────
  pyinstaller --windowed --onefile \\
      --name "TradingTerminal" \\
      --osx-bundle-identifier "com.quantdev.tradingterminal" \\
      --target-arch universal2 \\
      --hidden-import pandas_ta \\
      --hidden-import pandas_ta.momentum \\
      --hidden-import pandas_ta.trend \\
      --hidden-import pandas_ta.volatility \\
      --hidden-import yfinance \\
      --hidden-import pandas_datareader \\
      --hidden-import pandas_datareader.data \\
      --hidden-import matplotlib.backends.backend_qtagg \\
      --collect-data pandas_ta \\
      trading_terminal.py

  The finished .app bundle appears in dist/TradingTerminal.app
  Drag it to /Applications to install system-wide.

  ── Notes ─────────────────────────────────────────────────────────────────
  • Requires macOS 12 Monterey or later for full PyQt6 compatibility.
  • Apple Silicon Macs: the universal2 flag produces a fat binary that runs
    natively on both M-series and Intel without Rosetta.
  • All API calls are 100% free — no API keys required.
  • Trade logs are saved to: ~/Desktop/trading_history.csv
  • Virtual starting balance: $10,000 (no real money is involved).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ==============================================================================
# SECTION 1 — DEPENDENCIES & INITIALIZATION
# ==============================================================================

import os
import csv
import sys
import time
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Market data: Yahoo Finance ─────────────────────────────────────────────────
import yfinance as yf

# ── Technical indicators ───────────────────────────────────────────────────────
import pandas_ta as ta

# ── Macro data: FRED (Federal Reserve Economic Data) ──────────────────────────
try:
    from pandas_datareader import data as pdr
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# ── Matplotlib with the native Qt backend for canvas embedding ─────────────────
import matplotlib
matplotlib.use("QtAgg")                             # Must be set BEFORE any plt import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# ── PyQt6 GUI framework ────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QFrame, QSplitter, QPlainTextEdit, QGroupBox,
    QProgressBar, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QFontDatabase


# ─── Central Configuration ────────────────────────────────────────────────────

class Config:
    """All tunable application constants live here — change once, applied everywhere."""

    STARTING_BALANCE   : float = 10_000.0   # Virtual capital (USD)
    RISK_PER_TRADE     : float = 0.01        # 1% of account per trade
    MAX_DAILY_DRAWDOWN : float = 0.05        # 5% intraday circuit breaker
    ATR_PERIOD         : int   = 14
    ATR_MULT_SL        : float = 2.0         # Stop-loss  = 2× ATR
    ATR_MULT_TP        : float = 4.0         # Take-profit = 4× ATR
    EMA_FAST           : int   = 20
    EMA_MED            : int   = 50
    EMA_SLOW           : int   = 200
    RSI_PERIOD         : int   = 14
    VIX_REDUCE         : float = 25.0        # Halve position size above this VIX
    VIX_FREEZE         : float = 35.0        # Freeze all auto execution above this
    BACKTEST_YEARS     : int   = 2
    CSV_LOG_PATH       : str   = os.path.expanduser("~/Desktop/trading_history.csv")
    APP_NAME           : str   = "Cross-Asset Trading Terminal"
    VERSION            : str   = "1.0.0"


# ─── Design Tokens / Color Palette ───────────────────────────────────────────

class C:
    """Centralised dark-mode color palette — referenced throughout the stylesheet."""
    BG_DARK        = "#0D0F12"
    BG_MED         = "#141720"
    BG_CARD        = "#1A1D27"
    BG_INPUT       = "#20243A"
    BORDER         = "#272B40"
    BORDER_FOCUS   = "#3D7EFF"
    ACCENT_BLUE    = "#3D7EFF"
    ACCENT_GREEN   = "#00D68F"
    ACCENT_RED     = "#FF4757"
    ACCENT_AMBER   = "#FFA800"
    ACCENT_PURPLE  = "#7C4DFF"
    TXT_PRIMARY    = "#E8EAF6"
    TXT_SECONDARY  = "#8892A4"
    TXT_MUTED      = "#3E4460"
    CHART_BG       = "#0D0F12"
    CHART_GRID     = "#1A1E2C"


# ==============================================================================
# SECTION 2 — FREE GLOBAL MACRO DATA PIPELINE & LOCAL CSV LOGGER
# ==============================================================================

# Thread-safe lock protecting all CSV I/O
_csv_lock = threading.Lock()

CSV_COLUMNS = [
    "Timestamp", "Type", "Ticker", "Action",
    "Entry_Price", "Position_Size", "Stop_Loss",
    "Take_Profit", "Macro_VIX", "Status",
]


def _ensure_csv_header() -> None:
    """Create the trade log CSV with column headers if it does not yet exist."""
    path = Config.CSV_LOG_PATH
    if not os.path.exists(path):
        with _csv_lock:
            # Re-check after acquiring lock (double-checked locking pattern)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def log_trade(
    trade_type : str,
    ticker     : str,
    action     : str,
    entry      : float,
    size       : float,
    sl         : float,
    tp         : float,
    vix        : float,
    status     : str = "EXECUTED",
) -> None:
    """
    Append one trade execution row to the Desktop CSV log.
    Thread-safe — may be called from any QThread worker.
    """
    _ensure_csv_header()
    row = {
        "Timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type"         : trade_type,
        "Ticker"       : ticker,
        "Action"       : action,
        "Entry_Price"  : round(entry, 6),
        "Position_Size": round(size,  6),
        "Stop_Loss"    : round(sl,    6),
        "Take_Profit"  : round(tp,    6),
        "Macro_VIX"    : round(vix,   2),
        "Status"       : status,
    }
    with _csv_lock:
        with open(Config.CSV_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow(row)


# ── Free market & macro data fetchers ─────────────────────────────────────────

def _safe_yf_download(
    symbol: str, period: str = "5d", interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Download Yahoo Finance OHLCV; returns None on any error or empty result."""
    try:
        df = yf.download(
            symbol, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=False,
        )
        return df if (df is not None and not df.empty) else None
    except Exception:
        return None


def fetch_market_history(
    ticker: str, period: str = "2y", interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Full OHLCV history for backtesting or charting."""
    return _safe_yf_download(ticker, period=period, interval=interval)


def fetch_vix() -> float:
    """Current VIX close; defaults to 20.0 on failure."""
    df = _safe_yf_download("^VIX", period="5d")
    if df is not None and "Close" in df.columns:
        try:
            val = df["Close"].dropna().iloc[-1]
            return float(val)
        except Exception:
            pass
    return 20.0


def fetch_macro_snapshot() -> Dict:
    """
    Asynchronously gather six macro metrics via free public APIs:
      • VIX    — CBOE Volatility Index       (Yahoo Finance)
      • TNX    — US 10-Year Treasury Yield   (Yahoo Finance)
      • Gold   — Gold Futures (GC=F)         (Yahoo Finance)
      • Oil    — Brent Crude Futures (BZ=F)  (Yahoo Finance)
      • Fed    — US Federal Funds Rate       (FRED / pandas_datareader)
      • ECB    — ECB Deposit Facility Rate   (FRED / pandas_datareader)

    Degrades gracefully — any failed fetch keeps its default value.
    """
    snap = {
        "vix"     : 20.0,
        "tnx"     : 4.50,
        "gold"    : 2_300.0,
        "oil"     : 82.0,
        "fed_rate": 5.25,
        "ecb_rate": 4.00,
    }

    # ── Yahoo Finance tickers ──────────────────────────────────────────────────
    yf_map = {"vix": "^VIX", "tnx": "^TNX", "gold": "GC=F", "oil": "BZ=F"}
    for key, sym in yf_map.items():
        df = _safe_yf_download(sym, period="5d")
        if df is not None:
            close_col = "Close" if "Close" in df.columns else df.columns[0]
            try:
                snap[key] = float(df[close_col].dropna().iloc[-1])
            except Exception:
                pass

    # ── FRED macroeconomic series ──────────────────────────────────────────────
    if FRED_AVAILABLE:
        end   = datetime.now()
        start = end - timedelta(days=120)  # 4-month window ensures we get a data point
        fred_map = {
            "fed_rate": "FEDFUNDS",  # US Federal Funds Effective Rate
            "ecb_rate": "ECBDFR",   # ECB Deposit Facility Rate
        }
        for key, series_id in fred_map.items():
            try:
                df = pdr.DataReader(series_id, "fred", start, end)
                if df is not None and not df.empty:
                    snap[key] = float(df.iloc[-1, 0])
            except Exception:
                pass  # Silently fall back to default

    return snap


# ==============================================================================
# SECTION 3 — TECHNICAL ANALYSIS & AUTOMATED BACKTESTER
# ==============================================================================

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance DataFrames — collapse MultiIndex columns and
    lower-case all names so downstream code uses consistent keys.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the full indicator suite to an OHLCV DataFrame:
      EMA 20 / 50 / 200  •  RSI 14  •  MACD (12/26/9)  •  ATR 14

    Uses pandas_ta functional API for explicit, reproducible results.
    Returns a copy with NaN rows dropped.
    """
    df = _flatten_columns(df.copy())

    # Guard: ensure required columns exist
    required = {"close", "high", "low", "open"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"DataFrame missing columns. Got: {list(df.columns)}")

    # ── Trend: Exponential Moving Averages ────────────────────────────────────
    df["ema_20"]  = ta.ema(df["close"], length=Config.EMA_FAST)
    df["ema_50"]  = ta.ema(df["close"], length=Config.EMA_MED)
    df["ema_200"] = ta.ema(df["close"], length=Config.EMA_SLOW)

    # ── Momentum: Relative Strength Index ────────────────────────────────────
    df["rsi"] = ta.rsi(df["close"], length=Config.RSI_PERIOD)

    # ── Momentum: MACD  (returns DataFrame with 3 cols) ──────────────────────
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["macd"]        = macd_df.iloc[:, 0]  # MACD line
        df["macd_signal"] = macd_df.iloc[:, 2]  # Signal line
        df["macd_hist"]   = macd_df.iloc[:, 1]  # Histogram

    # ── Volatility: Average True Range ───────────────────────────────────────
    df["atr"] = ta.atr(df["high"], df["low"], df["close"],
                       length=Config.ATR_PERIOD)

    return df.dropna()


def generate_signal(row: pd.Series) -> str:
    """
    Rule-based directional signal evaluated on a single indicator row.

    Entry logic:
      BUY  → bullish EMA stack (20 > 50 > 200)  AND  MACD line > Signal line
             AND  MACD line > 0  AND  RSI < 70 (not overbought)
      SELL → bearish EMA stack (20 < 50 < 200)  AND  MACD line < Signal line
             AND  MACD line < 0  AND  RSI > 30 (not oversold)
      HOLD → all other conditions
    """
    ema_fast  = row.get("ema_20",  np.nan)
    ema_med   = row.get("ema_50",  np.nan)
    ema_slow  = row.get("ema_200", np.nan)
    rsi       = row.get("rsi",     50.0)
    macd      = row.get("macd",    0.0)
    macd_sig  = row.get("macd_signal", 0.0)

    if any(np.isnan(v) for v in [ema_fast, ema_med, ema_slow, rsi, macd, macd_sig]):
        return "HOLD"

    bull_ema     = ema_fast > ema_med > ema_slow
    bear_ema     = ema_fast < ema_med < ema_slow
    macd_bull    = macd > macd_sig and macd > 0
    macd_bear    = macd < macd_sig and macd < 0
    rsi_ok_bull  = rsi < 70
    rsi_ok_bear  = rsi > 30

    if bull_ema and macd_bull and rsi_ok_bull:
        return "BUY"
    if bear_ema and macd_bear and rsi_ok_bear:
        return "SELL"
    return "HOLD"


# ── Backtest result container ─────────────────────────────────────────────────

class BacktestResult:
    """All outputs produced by the backtesting engine."""
    __slots__ = (
        "ticker", "price_series", "indicator_df",
        "equity_curve", "trades",
        "final_balance", "total_return", "max_drawdown",
        "win_rate", "total_trades", "profit_factor", "sharpe",
        "error",
    )

    def __init__(self) -> None:
        self.ticker        : str              = ""
        self.price_series  : pd.Series        = pd.Series(dtype=float)
        self.indicator_df  : pd.DataFrame     = pd.DataFrame()
        self.equity_curve  : pd.Series        = pd.Series(dtype=float)
        self.trades        : List[Dict]       = []
        self.final_balance : float            = Config.STARTING_BALANCE
        self.total_return  : float            = 0.0
        self.max_drawdown  : float            = 0.0
        self.win_rate      : float            = 0.0
        self.total_trades  : int              = 0
        self.profit_factor : float            = 0.0
        self.sharpe        : float            = 0.0
        self.error         : Optional[str]    = None


def run_backtest(ticker: str, vix_level: float = 20.0) -> BacktestResult:
    """
    Full event-driven backtest over 2 years of daily OHLCV data.

    Risk framework enforced on every entry:
      • Position size  = (Balance × 1%) / (2 × ATR)
      • Stop-loss      = 2 × ATR from entry
      • Take-profit    = 4 × ATR from entry  (2 : 1 reward/risk minimum)
      • VIX > 25       → cut risk_amount by 50%
      • VIX > 35       → skip all new entries (freeze gate)
      • Daily drawdown ≥ 5% → halt trading for remainder of that day

    One position at a time; exits are evaluated on the close of each bar.
    """
    result = BacktestResult()
    result.ticker = ticker

    # ── 1. Fetch & validate data ───────────────────────────────────────────────
    df_raw = fetch_market_history(ticker, period="2y", interval="1d")
    if df_raw is None or len(df_raw) < 220:
        result.error = f"Insufficient history for {ticker} (need ≥ 220 bars)."
        return result

    try:
        df = compute_indicators(df_raw)
    except Exception as exc:
        result.error = f"Indicator computation failed: {exc}"
        return result

    result.price_series = df["close"].copy()
    result.indicator_df = df.copy()

    # ── 2. Simulation loop ─────────────────────────────────────────────────────
    balance         = Config.STARTING_BALANCE
    peak_balance    = balance
    daily_open_bal  = balance
    last_date       = None
    position        = None    # Active trade dict or None
    equity_points   = {}      # {DatetimeIndex → balance}
    trades          = []

    for i, (idx, row) in enumerate(df.iterrows()):
        price    = float(row["close"])
        atr      = float(row["atr"])
        cur_date = str(idx)[:10]

        # ── Reset daily drawdown tracker at each new calendar day ──────────────
        if cur_date != last_date:
            daily_open_bal = balance
            last_date = cur_date

        # ── 5% daily drawdown circuit breaker ─────────────────────────────────
        daily_dd = (daily_open_bal - balance) / daily_open_bal if daily_open_bal > 0 else 0
        if daily_dd >= Config.MAX_DAILY_DRAWDOWN:
            equity_points[idx] = balance
            continue

        # ── VIX hard freeze gate ───────────────────────────────────────────────
        if vix_level > Config.VIX_FREEZE:
            equity_points[idx] = balance
            continue

        # ── Manage open position (mark-to-close exit check) ───────────────────
        if position is not None:
            is_long  = position["action"] == "BUY"
            hit_sl = price <= position["sl"] if is_long else price >= position["sl"]
            hit_tp = price >= position["tp"] if is_long else price <= position["tp"]

            if hit_sl or hit_tp:
                # Determine exact exit price (conservative: use SL/TP level)
                exit_price = position["tp"] if hit_tp else position["sl"]
                direction  = 1 if is_long else -1
                pnl        = (exit_price - position["entry"]) * direction * position["size"]
                balance   += pnl
                peak_balance = max(peak_balance, balance)

                trades.append({
                    **position,
                    "exit_price": exit_price,
                    "exit_date" : cur_date,
                    "pnl"       : round(pnl, 4),
                    "outcome"   : "WIN" if pnl > 0 else "LOSS",
                })
                position = None

        # ── Entry logic (only after EMA-200 warmup period) ────────────────────
        if position is None and i >= Config.EMA_SLOW and atr > 0:
            signal = generate_signal(row)

            if signal in ("BUY", "SELL"):
                # Position sizing: 1% risk / SL distance per unit
                risk_usd = balance * Config.RISK_PER_TRADE
                if vix_level > Config.VIX_REDUCE:
                    risk_usd *= 0.50  # Macro VIX gate: halve size

                sl_dist   = Config.ATR_MULT_SL * atr
                tp_dist   = Config.ATR_MULT_TP * atr
                size      = risk_usd / sl_dist if sl_dist > 0 else 0

                if size > 0:
                    is_long = signal == "BUY"
                    position = {
                        "ticker"     : ticker,
                        "action"     : signal,
                        "entry_date" : cur_date,
                        "entry"      : price,
                        "sl"         : price - sl_dist if is_long else price + sl_dist,
                        "tp"         : price + tp_dist if is_long else price - tp_dist,
                        "size"       : size,
                        "atr"        : atr,
                        "vix"        : vix_level,
                    }

        equity_points[idx] = balance

    # ── Force-close any position still open at the last bar ───────────────────
    if position is not None:
        last_price = float(df["close"].iloc[-1])
        direction  = 1 if position["action"] == "BUY" else -1
        pnl        = (last_price - position["entry"]) * direction * position["size"]
        balance   += pnl
        trades.append({
            **position,
            "exit_price": last_price,
            "exit_date" : str(df.index[-1])[:10],
            "pnl"       : round(pnl, 4),
            "outcome"   : "WIN" if pnl > 0 else "LOSS",
        })

    # ── 3. Compute performance statistics ─────────────────────────────────────
    equity = pd.Series(equity_points)
    result.equity_curve  = equity
    result.trades        = trades
    result.final_balance = round(balance, 2)
    result.total_return  = round((balance / Config.STARTING_BALANCE - 1) * 100, 2)
    result.total_trades  = len(trades)

    wins  = [t for t in trades if t.get("outcome") == "WIN"]
    losses = [t for t in trades if t.get("outcome") == "LOSS"]
    result.win_rate = round(len(wins) / max(len(trades), 1) * 100, 1)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    result.profit_factor = round(
        gross_profit / gross_loss if gross_loss > 0 else float("inf"), 2)

    # Max drawdown from equity curve
    roll_max   = equity.cummax()
    dd_series  = (equity - roll_max) / roll_max
    result.max_drawdown = round(float(dd_series.min()) * 100, 2)

    # Annualised Sharpe ratio (252 trading-day convention)
    daily_rets = equity.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        result.sharpe = round(
            (daily_rets.mean() / daily_rets.std()) * (252 ** 0.5), 2)

    return result


# ==============================================================================
# SECTION 4 — DARK-MODE STYLESHEET
# ==============================================================================

DARK_QSS = f"""
/* ── Global base ─────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {C.BG_DARK};
    color: {C.TXT_PRIMARY};
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

/* ── Cards (named frames) ────────────────────────────────────────────── */
QFrame#card {{
    background-color: {C.BG_CARD};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
}}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {C.BG_CARD};
    border: 1px solid {C.BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 10px;
    font-size: 10px;
    font-weight: 700;
    color: {C.TXT_SECONDARY};
    letter-spacing: 1.2px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
}}

/* ── Primary buttons ─────────────────────────────────────────────────── */
QPushButton {{
    background-color: {C.ACCENT_BLUE};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}
QPushButton:hover  {{ background-color: #5B95FF; }}
QPushButton:pressed {{ background-color: #2A5ECC; }}
QPushButton:disabled {{
    background-color: {C.BG_INPUT};
    color: {C.TXT_MUTED};
}}

/* ── Semantic button variants ────────────────────────────────────────── */
QPushButton#success {{
    background-color: {C.ACCENT_GREEN};
    color: {C.BG_DARK};
}}
QPushButton#success:hover  {{ background-color: #00F0A0; }}

QPushButton#warning {{
    background-color: {C.ACCENT_AMBER};
    color: {C.BG_DARK};
}}
QPushButton#warning:hover  {{ background-color: #FFB733; }}

QPushButton#ghost {{
    background-color: transparent;
    color: {C.TXT_SECONDARY};
    border: 1px solid {C.BORDER};
}}
QPushButton#ghost:hover {{ background-color: {C.BG_CARD}; color: {C.TXT_PRIMARY}; }}

/* ── Text inputs ─────────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {C.BG_INPUT};
    border: 1px solid {C.BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    color: {C.TXT_PRIMARY};
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {C.BORDER_FOCUS};
    background-color: #252A42;
}}

/* ── Combo boxes ─────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {C.BG_INPUT};
    border: 1px solid {C.BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    color: {C.TXT_PRIMARY};
    font-size: 13px;
    min-width: 110px;
}}
QComboBox:focus {{ border-color: {C.BORDER_FOCUS}; }}
QComboBox::drop-down {{ border: none; width: 30px; }}
QComboBox::down-arrow {{
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {C.TXT_SECONDARY};
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {C.BG_CARD};
    border: 1px solid {C.BORDER};
    border-radius: 6px;
    color: {C.TXT_PRIMARY};
    selection-background-color: {C.ACCENT_BLUE};
    outline: none;
}}

/* ── Terminal log ────────────────────────────────────────────────────── */
QPlainTextEdit {{
    background-color: #090B0E;
    border: 1px solid {C.BORDER};
    border-radius: 8px;
    color: #6FD67A;
    font-family: "SF Mono", Menlo, "Courier New", monospace;
    font-size: 11px;
    padding: 10px 12px;
    line-height: 1.5;
}}

/* ── Splitter ─────────────────────────────────────────────────────────── */
QSplitter::handle {{ background-color: {C.BORDER}; width: 1px; height: 1px; }}

/* ── Scrollbars ──────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {C.BG_MED};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C.BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {C.BG_INPUT};
    border: none;
    border-radius: 3px;
    height: 5px;
}}
QProgressBar::chunk {{
    background-color: {C.ACCENT_BLUE};
    border-radius: 3px;
}}

/* ── Tooltips ────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {C.BG_CARD};
    border: 1px solid {C.BORDER};
    color: {C.TXT_PRIMARY};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}
"""


# ==============================================================================
# SECTION 4 (continued) — REUSABLE WIDGET COMPONENTS
# ==============================================================================

class KPICard(QFrame):
    """Single metric tile: small upper label + large coloured value."""

    def __init__(
        self,
        label  : str,
        value  : str = "—",
        colour : str = C.ACCENT_BLUE,
        parent = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumHeight(78)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)

        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            f"color:{C.TXT_SECONDARY}; font-size:9px; font-weight:700;"
            f"letter-spacing:1.1px; background:transparent; border:none;")

        self._val = QLabel(value)
        self._val.setStyleSheet(
            f"color:{colour}; font-size:22px; font-weight:700;"
            f"background:transparent; border:none;")

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)

    def set_value(self, value: str, colour: Optional[str] = None) -> None:
        self._val.setText(value)
        if colour:
            self._val.setStyleSheet(
                f"color:{colour}; font-size:22px; font-weight:700;"
                f"background:transparent; border:none;")


class StatPill(QWidget):
    """Compact stat used in the backtest results bar above the chart."""

    def __init__(self, label: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color:{C.TXT_MUTED}; font-size:9px; font-weight:700; letter-spacing:0.9px;")

        self._val = QLabel(value)
        self._val.setStyleSheet(
            f"color:{C.TXT_PRIMARY}; font-size:13px; font-weight:700;")

        layout.addWidget(lbl)
        layout.addWidget(self._val)

    def set_value(self, value: str, colour: Optional[str] = None) -> None:
        self._val.setText(value)
        base = "font-size:13px; font-weight:700;"
        self._val.setStyleSheet(
            f"color:{colour if colour else C.TXT_PRIMARY}; {base}")


class TerminalLog(QPlainTextEdit):
    """
    Rolling monospace terminal window with colour-coded log levels.
    Auto-scrolls to the latest entry; capped at MAX_LINES to prevent bloat.
    """

    MAX_LINES = 400

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setPlaceholderText("System terminal ready…")

    def append_line(self, text: str, level: str = "INFO") -> None:
        level_colours = {
            "INFO"   : "#5DB8FF",
            "SUCCESS": "#00D68F",
            "WARN"   : "#FFA800",
            "ERROR"  : "#FF4757",
            "TRADE"  : "#B44DFF",
        }
        ts    = datetime.now().strftime("%H:%M:%S")
        clr   = level_colours.get(level, "#5DB8FF")
        # Use HTML for colour but keep font from the stylesheet
        html  = (
            f'<span style="color:{C.TXT_MUTED}">[{ts}]</span> '
            f'<span style="color:{clr}; font-weight:600;">[{level:7s}]</span> '
            f'<span style="color:#C8D4E8;">{text}</span>'
        )
        self.appendHtml(html)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ── Embedded Matplotlib chart canvas ─────────────────────────────────────────

class ChartCanvas(FigureCanvas):
    """
    Matplotlib figure embedded inside a PyQt6 QWidget via the QtAgg backend.
    Renders a 2-panel layout:
      Top  — asset price history with EMA overlays
      Bottom — backtest equity curve vs. starting balance baseline
    """

    def __init__(self, parent=None) -> None:
        self.fig = Figure(figsize=(11, 8), facecolor=C.CHART_BG, dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._render_placeholder()

    # ── Shared axis styling ───────────────────────────────────────────────────
    def _style_ax(self, ax, title: str = "") -> None:
        ax.set_facecolor(C.CHART_BG)
        ax.tick_params(colors=C.TXT_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(C.TXT_SECONDARY)
        ax.yaxis.label.set_color(C.TXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(C.CHART_GRID)
        ax.grid(True, color=C.CHART_GRID, linestyle="--",
                linewidth=0.5, alpha=0.6)
        if title:
            ax.set_title(title, color=C.TXT_PRIMARY, fontsize=11,
                         fontweight="bold", pad=10)

    def _render_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_ax(ax)
        ax.text(
            0.5, 0.5,
            "Enter a ticker symbol above\nand click  ▶  Run Analytics & Backtest",
            ha="center", va="center", transform=ax.transAxes,
            color=C.TXT_SECONDARY, fontsize=14, alpha=0.5,
            linespacing=1.8,
        )
        self.fig.tight_layout(pad=2)
        self.draw()

    def plot_results(self, result: BacktestResult) -> None:
        """Render the full dual-panel chart from a BacktestResult object."""
        self.fig.clear()
        gs = GridSpec(2, 1, figure=self.fig,
                      height_ratios=[2.8, 1.8], hspace=0.40)

        df = result.indicator_df

        # ── Top panel: price + EMA overlays ───────────────────────────────────
        ax1 = self.fig.add_subplot(gs[0])
        self._style_ax(ax1, f"{result.ticker}  —  Price & EMA Stack")

        price = df["close"]
        ax1.plot(price.index, price.values,
                 color=C.TXT_PRIMARY, linewidth=1.1, alpha=0.9, label="Close")

        ema_specs = [
            ("ema_20",  C.ACCENT_BLUE,   "EMA 20"),
            ("ema_50",  C.ACCENT_AMBER,  "EMA 50"),
            ("ema_200", C.ACCENT_RED,    "EMA 200"),
        ]
        for col, colour, lbl in ema_specs:
            if col in df.columns:
                ax1.plot(df.index, df[col].values,
                         color=colour, linewidth=1.0, alpha=0.75, label=lbl)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")
        ax1.set_ylabel("Price (USD)", color=C.TXT_SECONDARY, fontsize=10)
        ax1.legend(
            loc="upper left", facecolor=C.BG_CARD, edgecolor=C.BORDER,
            labelcolor=C.TXT_PRIMARY, fontsize=8, framealpha=0.8,
        )

        # ── Bottom panel: equity curve ─────────────────────────────────────────
        ax2 = self.fig.add_subplot(gs[1])
        self._style_ax(ax2, f"Backtest Equity Curve  (start ${Config.STARTING_BALANCE:,})")

        eq      = result.equity_curve
        ret_pos = result.total_return >= 0
        curve_colour = C.ACCENT_GREEN if ret_pos else C.ACCENT_RED

        ax2.plot(eq.index, eq.values, color=curve_colour,
                 linewidth=1.6, label=f"Equity  ({result.total_return:+.2f}%)")
        ax2.fill_between(
            eq.index, Config.STARTING_BALANCE, eq.values,
            alpha=0.12, color=curve_colour,
        )
        ax2.axhline(
            Config.STARTING_BALANCE, color=C.TXT_MUTED,
            linestyle="--", linewidth=0.8, alpha=0.7, label="Start ($10,000)",
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")
        ax2.set_ylabel("Balance (USD)", color=C.TXT_SECONDARY, fontsize=10)
        ax2.legend(
            loc="upper left", facecolor=C.BG_CARD, edgecolor=C.BORDER,
            labelcolor=C.TXT_PRIMARY, fontsize=8, framealpha=0.8,
        )

        self.fig.tight_layout(pad=2.2)
        self.draw()


# ==============================================================================
# SECTION 5 — ASYNCHRONOUS QThread WORKERS
# ==============================================================================

class MacroDataWorker(QThread):
    """Fetch the full macro snapshot without blocking the GUI event loop."""

    data_ready = pyqtSignal(dict)
    log_msg    = pyqtSignal(str, str)   # (message, level)

    def run(self) -> None:
        self.log_msg.emit(
            "Fetching global macro data  (VIX · TNX · Gold · Oil · FRED)…", "INFO")
        try:
            snap = fetch_macro_snapshot()
            self.data_ready.emit(snap)
            self.log_msg.emit(
                f"Macro snapshot received  ▸  "
                f"VIX {snap['vix']:.1f}  |  "
                f"10Y {snap['tnx']:.2f}%  |  "
                f"Gold ${snap['gold']:,.0f}  |  "
                f"Brent ${snap['oil']:.1f}  |  "
                f"Fed {snap['fed_rate']:.2f}%  |  "
                f"ECB {snap['ecb_rate']:.2f}%",
                "SUCCESS",
            )
        except Exception as exc:
            self.log_msg.emit(f"Macro pipeline error: {exc}", "ERROR")
            self.data_ready.emit({})


class BacktestWorker(QThread):
    """Run the full 2-year backtest simulation in a background thread."""

    result_ready = pyqtSignal(object)   # BacktestResult
    log_msg      = pyqtSignal(str, str)
    progress     = pyqtSignal(int)

    def __init__(self, ticker: str, vix: float, parent=None) -> None:
        super().__init__(parent)
        self.ticker = ticker
        self.vix    = vix

    def run(self) -> None:
        self.log_msg.emit(
            f"Backtest engine starting  ▸  {self.ticker}  |  VIX={self.vix:.1f}", "INFO")
        self.progress.emit(15)
        try:
            result = run_backtest(self.ticker, self.vix)
            self.progress.emit(85)
            if result.error:
                self.log_msg.emit(f"Backtest halted: {result.error}", "ERROR")
            else:
                sign = "+" if result.total_return >= 0 else ""
                self.log_msg.emit(
                    f"Backtest complete  ▸  "
                    f"Return {sign}{result.total_return:.2f}%  |  "
                    f"Win {result.win_rate:.1f}%  |  "
                    f"PF {result.profit_factor:.2f}  |  "
                    f"MaxDD {result.max_drawdown:.2f}%  |  "
                    f"Sharpe {result.sharpe:.2f}  |  "
                    f"{result.total_trades} trades",
                    "SUCCESS",
                )
            self.result_ready.emit(result)
        except Exception:
            self.log_msg.emit(
                f"Backtest exception:\n{traceback.format_exc()}", "ERROR")
            r = BacktestResult()
            r.error = "Unhandled exception — see terminal."
            self.result_ready.emit(r)
        self.progress.emit(100)


class SignalWorker(QThread):
    """
    Generate the current directional signal and position parameters
    for the automated execution engine card.
    """

    signal_ready = pyqtSignal(str, float, float, float, float)
    # (signal, entry, sl, tp, size)
    log_msg = pyqtSignal(str, str)

    def __init__(self, ticker: str, balance: float, vix: float,
                 parent=None) -> None:
        super().__init__(parent)
        self.ticker  = ticker
        self.balance = balance
        self.vix     = vix

    def run(self) -> None:
        self.log_msg.emit(
            f"Generating live signal  ▸  {self.ticker}  (VIX={self.vix:.1f})", "INFO")
        try:
            if self.vix > Config.VIX_FREEZE:
                self.log_msg.emit(
                    f"VIX={self.vix:.1f} exceeds freeze threshold ({Config.VIX_FREEZE}).  "
                    f"Auto execution SUSPENDED.", "ERROR")
                self.signal_ready.emit("HOLD", 0, 0, 0, 0)
                return

            df = fetch_market_history(self.ticker, period="1y", interval="1d")
            if df is None or len(df) < 60:
                self.log_msg.emit("Not enough data to compute live signal.", "WARN")
                self.signal_ready.emit("HOLD", 0, 0, 0, 0)
                return

            df   = compute_indicators(df)
            row  = df.iloc[-1]
            sig  = generate_signal(row)
            entry = float(row["close"])
            atr   = float(row["atr"])

            risk_usd = self.balance * Config.RISK_PER_TRADE
            if self.vix > Config.VIX_REDUCE:
                risk_usd *= 0.50
                self.log_msg.emit(
                    f"VIX={self.vix:.1f} > {Config.VIX_REDUCE}  ▸  "
                    f"Position size reduced 50% (macro volatility gate).", "WARN")

            sl_dist = Config.ATR_MULT_SL * atr
            tp_dist = Config.ATR_MULT_TP * atr
            size    = risk_usd / sl_dist if sl_dist > 0 else 0.0
            is_long = sig == "BUY"
            sl      = entry - sl_dist if is_long else entry + sl_dist
            tp      = entry + tp_dist if is_long else entry - tp_dist

            self.log_msg.emit(
                f"Signal computed  ▸  {sig}  {self.ticker}  @  ${entry:.4f}  |  "
                f"SL ${sl:.4f}  |  TP ${tp:.4f}  |  "
                f"Size {size:.4f} units  (ATR={atr:.4f})",
                "TRADE",
            )
            self.signal_ready.emit(sig, entry, sl, tp, size)

        except Exception as exc:
            self.log_msg.emit(f"Signal error: {exc}", "ERROR")
            self.signal_ready.emit("HOLD", 0, 0, 0, 0)


class ManualTradeWorker(QThread):
    """
    Validate user-entered trade parameters, apply 1% risk sizing,
    and write the confirmed trade to the Desktop CSV log.
    """

    trade_done = pyqtSignal(bool, str, float)   # (success, message, computed_size)
    log_msg    = pyqtSignal(str, str)

    def __init__(self, ticker: str, action: str, entry: float,
                 sl: float, tp: float, balance: float, vix: float,
                 parent=None) -> None:
        super().__init__(parent)
        self.ticker  = ticker
        self.action  = action
        self.entry   = entry
        self.sl      = sl
        self.tp      = tp
        self.balance = balance
        self.vix     = vix

    def run(self) -> None:
        try:
            # ── Directional sanity checks ──────────────────────────────────────
            if self.action == "BUY":
                if self.sl >= self.entry:
                    self.trade_done.emit(
                        False, "BUY stop-loss must be below entry price.", 0.0)
                    return
                if self.tp <= self.entry:
                    self.trade_done.emit(
                        False, "BUY take-profit must be above entry price.", 0.0)
                    return
            else:
                if self.sl <= self.entry:
                    self.trade_done.emit(
                        False, "SELL stop-loss must be above entry price.", 0.0)
                    return
                if self.tp >= self.entry:
                    self.trade_done.emit(
                        False, "SELL take-profit must be below entry price.", 0.0)
                    return

            # ── 1% risk position sizing (same rule as automated engine) ────────
            risk_usd = self.balance * Config.RISK_PER_TRADE
            sl_dist  = abs(self.entry - self.sl)
            if sl_dist <= 0:
                self.trade_done.emit(False, "Entry and SL cannot be equal.", 0.0)
                return

            size = risk_usd / sl_dist

            # ── Log to Desktop CSV ─────────────────────────────────────────────
            log_trade(
                trade_type="Manual",
                ticker=self.ticker,
                action=self.action,
                entry=self.entry,
                size=size,
                sl=self.sl,
                tp=self.tp,
                vix=self.vix,
            )

            msg = (
                f"Manual {self.action} {self.ticker}  |  "
                f"Entry ${self.entry:.4f}  |  "
                f"SL ${self.sl:.4f}  |  "
                f"TP ${self.tp:.4f}  |  "
                f"Size {size:.4f} units  (${risk_usd:.2f} risk)"
            )
            self.log_msg.emit(msg, "TRADE")
            self.log_msg.emit(
                f"Saved to Desktop CSV  ▸  {Config.CSV_LOG_PATH}", "SUCCESS")
            self.trade_done.emit(True, msg, size)

        except Exception as exc:
            self.log_msg.emit(f"Manual trade error: {exc}", "ERROR")
            self.trade_done.emit(False, str(exc), 0.0)


# ==============================================================================
# SECTION 4 (continued) — MAIN APPLICATION WINDOW
# ==============================================================================

class MainWindow(QMainWindow):
    """
    Top-level macOS application window.

    Layout (vertical stack):
      ┌─────────────────────────────────────────────────────────┐
      │  TOP BAR  — ticker / asset-class / action buttons       │
      ├──────────────────────┬──────────────────────────────────┤
      │  LEFT PANEL          │  RIGHT PANEL                     │
      │  • KPI cards         │  • Stats bar                     │
      │  • Auto signal card  │  • Embedded Matplotlib chart     │
      │  • Manual trade form │                                  │
      ├──────────────────────┴──────────────────────────────────┤
      │  BOTTOM BAR — rolling terminal log                      │
      └─────────────────────────────────────────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{Config.APP_NAME}  ·  v{Config.VERSION}")
        self.setMinimumSize(1280, 820)
        self.resize(1460, 940)

        # ── Internal state ─────────────────────────────────────────────────────
        self._balance    : float          = Config.STARTING_BALANCE
        self._vix        : float          = 20.0
        self._macro_snap : Dict           = {}
        self._auto_data  : Dict           = {}   # live signal params
        self._workers    : List[QThread]  = []   # keep alive refs to all threads

        # ── Build layout ───────────────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._build_top_bar())
        vbox.addWidget(self._build_workspace(), stretch=1)
        vbox.addWidget(self._build_bottom_bar())

        # ── Boot sequence ──────────────────────────────────────────────────────
        self._terminal.append_line(
            f"{Config.APP_NAME}  v{Config.VERSION}  |  "
            f"Virtual balance ${Config.STARTING_BALANCE:,.2f}  |  "
            f"macOS native build  (PyQt6 + QtAgg + pandas_ta)",
            "SUCCESS",
        )
        self._terminal.append_line(
            f"Trade log path: {Config.CSV_LOG_PATH}", "INFO")
        QTimer.singleShot(500, self._fetch_macro)   # slight delay for UI render

    # ══════════════════════════════════════════════════════════════
    # UI BUILDERS
    # ══════════════════════════════════════════════════════════════

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(74)
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {C.BG_MED};
                border: none;
                border-bottom: 1px solid {C.BORDER};
            }}
        """)
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 0, 20, 0)
        h.setSpacing(12)

        # ── Branding block ─────────────────────────────────────────────────────
        brand = QVBoxLayout()
        brand.setSpacing(1)

        title = QLabel(Config.APP_NAME)
        title.setStyleSheet(
            f"color:{C.TXT_PRIMARY}; font-size:16px; font-weight:700;"
            f"background:transparent; border:none;")

        sub = QLabel(
            "Quantitative Multi-Asset Terminal  ·  100% Free APIs  ·  "
            "Virtual $10,000 Framework")
        sub.setStyleSheet(
            f"color:{C.TXT_SECONDARY}; font-size:11px;"
            f"background:transparent; border:none;")

        brand.addWidget(title)
        brand.addWidget(sub)
        h.addLayout(brand)
        h.addStretch()

        # ── Ticker input ───────────────────────────────────────────────────────
        h.addWidget(self._muted_label("Symbol"))
        self._ticker_input = QLineEdit("SPY")
        self._ticker_input.setFixedWidth(100)
        self._ticker_input.setToolTip(
            "Any Yahoo Finance symbol — e.g. AAPL  BTC-USD  GC=F  ^GSPC  EURUSD=X")
        h.addWidget(self._ticker_input)

        h.addWidget(self._muted_label("Asset Class"))
        self._asset_combo = QComboBox()
        self._asset_combo.addItems([
            "Equities", "ETFs", "Crypto", "Forex",
            "Commodities", "Indices", "Futures",
        ])
        self._asset_combo.setFixedWidth(130)
        h.addWidget(self._asset_combo)

        # ── Primary action button ──────────────────────────────────────────────
        self._run_btn = QPushButton("▶   Run Analytics & Backtest")
        self._run_btn.setFixedHeight(46)
        self._run_btn.setMinimumWidth(230)
        self._run_btn.clicked.connect(self._on_run)
        h.addWidget(self._run_btn)

        # ── Refresh macro button ───────────────────────────────────────────────
        self._macro_btn = QPushButton("⟳   Refresh Macro")
        self._macro_btn.setFixedHeight(46)
        self._macro_btn.setObjectName("ghost")
        self._macro_btn.clicked.connect(self._fetch_macro)
        h.addWidget(self._macro_btn)

        return bar

    def _build_workspace(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(14, 14, 14, 8)
        splitter.setHandleWidth(1)

        # ── Left panel ─────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(470)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 10, 0)
        lv.setSpacing(10)

        lv.addWidget(self._build_kpi_group())
        lv.addWidget(self._build_auto_group())
        lv.addWidget(self._build_manual_group())
        lv.addStretch()

        # ── Right panel ────────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 0, 0, 0)
        rv.setSpacing(0)

        rv.addWidget(self._build_stats_bar())

        self._chart = ChartCanvas()
        rv.addWidget(self._chart, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_kpi_group(self) -> QGroupBox:
        box = QGroupBox("Live Status")
        grid = QGridLayout(box)
        grid.setContentsMargins(12, 18, 12, 12)
        grid.setSpacing(8)

        self._kpi_balance   = KPICard("Account Balance",
                                      f"${self._balance:,.2f}", C.ACCENT_GREEN)
        self._kpi_vix       = KPICard("CBOE VIX",
                                      f"{self._vix:.1f}", C.ACCENT_AMBER)
        self._kpi_risk_level= KPICard("Macro Risk Level",
                                      "NORMAL", C.ACCENT_GREEN)
        self._kpi_fed       = KPICard("Fed Funds Rate", "—%", C.ACCENT_BLUE)

        grid.addWidget(self._kpi_balance,    0, 0)
        grid.addWidget(self._kpi_vix,        0, 1)
        grid.addWidget(self._kpi_risk_level, 1, 0)
        grid.addWidget(self._kpi_fed,        1, 1)
        return box

    def _build_auto_group(self) -> QGroupBox:
        box = QGroupBox("Automated Signal Engine")
        v = QVBoxLayout(box)
        v.setContentsMargins(14, 18, 14, 14)
        v.setSpacing(10)

        # ── Signal badge row ───────────────────────────────────────────────────
        sig_row = QHBoxLayout()
        sig_row.addWidget(
            self._label("Current Signal:", C.TXT_SECONDARY, 11))

        self._signal_badge = QLabel("ANALYZING…")
        self._signal_badge.setStyleSheet(
            f"color:{C.TXT_MUTED}; font-size:20px; font-weight:800;"
            f"background:transparent; border:none; letter-spacing:0.5px;")
        sig_row.addWidget(self._signal_badge)
        sig_row.addStretch()
        v.addLayout(sig_row)

        # ── Signal parameter grid ──────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._auto_entry = self._label("—", C.TXT_PRIMARY, 13, bold=True)
        self._auto_sl    = self._label("—", C.ACCENT_RED,  13, bold=True)
        self._auto_tp    = self._label("—", C.ACCENT_GREEN,13, bold=True)
        self._auto_size  = self._label("—", C.ACCENT_BLUE, 13, bold=True)
        self._auto_atr   = self._label("—", C.TXT_SECONDARY,12)

        for lbl_txt, widget in [
            ("Entry:",         self._auto_entry),
            ("Stop-Loss:",     self._auto_sl),
            ("Take-Profit:",   self._auto_tp),
            ("Position Size:", self._auto_size),
            ("ATR (14):",      self._auto_atr),
        ]:
            form.addRow(self._label(lbl_txt, C.TXT_SECONDARY, 11), widget)
        v.addLayout(form)

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._refresh_sig_btn = QPushButton("⟳  Refresh Signal")
        self._refresh_sig_btn.setObjectName("warning")
        self._refresh_sig_btn.clicked.connect(self._on_refresh_signal)

        self._commit_btn = QPushButton("✓  Commit Auto Trade")
        self._commit_btn.setObjectName("success")
        self._commit_btn.setEnabled(False)
        self._commit_btn.setToolTip(
            "Logs the auto-generated trade to your Desktop CSV.\n"
            "Disabled for HOLD signals or when VIX is above freeze threshold.")
        self._commit_btn.clicked.connect(self._on_commit_auto)

        btn_row.addWidget(self._refresh_sig_btn)
        btn_row.addWidget(self._commit_btn)
        v.addLayout(btn_row)
        return box

    def _build_manual_group(self) -> QGroupBox:
        box = QGroupBox("Manual Trade Builder  ·  Override")
        form = QFormLayout(box)
        form.setContentsMargins(14, 18, 14, 14)
        form.setSpacing(9)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def fl(txt: str) -> QLabel:
            return self._label(txt, C.TXT_SECONDARY, 11)

        self._m_ticker = QLineEdit()
        self._m_ticker.setPlaceholderText("AAPL  BTC-USD  GC=F  EURUSD=X…")

        self._m_action = QComboBox()
        self._m_action.addItems(["BUY", "SELL"])

        self._m_entry = QLineEdit()
        self._m_entry.setPlaceholderText("Entry price (USD)")

        self._m_sl = QLineEdit()
        self._m_sl.setPlaceholderText("Stop-loss price")

        self._m_tp = QLineEdit()
        self._m_tp.setPlaceholderText("Take-profit price")

        # Live size preview — updates as user types
        self._m_size_preview = QLabel("Position size: —")
        self._m_size_preview.setStyleSheet(
            f"color:{C.ACCENT_BLUE}; font-size:12px; font-weight:700;"
            f"background:transparent; border:none;")

        self._m_entry.textChanged.connect(self._update_size_preview)
        self._m_sl.textChanged.connect(self._update_size_preview)

        form.addRow(fl("Symbol:"),      self._m_ticker)
        form.addRow(fl("Direction:"),   self._m_action)
        form.addRow(fl("Entry Price:"), self._m_entry)
        form.addRow(fl("Stop-Loss:"),   self._m_sl)
        form.addRow(fl("Take-Profit:"), self._m_tp)
        form.addRow(self._m_size_preview)

        exec_btn = QPushButton("⚡  Calculate Size & Execute Trade")
        exec_btn.setObjectName("success")
        exec_btn.setToolTip(
            "Validates your inputs, applies 1% risk sizing, "
            "and appends the trade to ~/Desktop/trading_history.csv")
        exec_btn.clicked.connect(self._on_manual_execute)
        form.addRow(exec_btn)
        return box

    def _build_stats_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: {C.BG_MED};
                border-bottom: 1px solid {C.BORDER};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
        """)
        h = QHBoxLayout(bar)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(4)

        self._stat_return  = StatPill("Return")
        self._stat_winrate = StatPill("Win Rate")
        self._stat_maxdd   = StatPill("Max Drawdown")
        self._stat_sharpe  = StatPill("Sharpe")
        self._stat_pf      = StatPill("Profit Factor")
        self._stat_trades  = StatPill("Trades")
        self._stat_final   = StatPill("Final Balance")

        for pill in [self._stat_return, self._stat_winrate, self._stat_maxdd,
                     self._stat_sharpe, self._stat_pf, self._stat_trades,
                     self._stat_final]:
            h.addWidget(pill)
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setStyleSheet(f"color:{C.BORDER}; max-width:1px;")
            h.addWidget(sep)

        h.addStretch()

        # Progress bar shown only during backtest
        self._progress = QProgressBar()
        self._progress.setFixedWidth(130)
        self._progress.setFixedHeight(5)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        h.addWidget(self._progress)
        return bar

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(188)
        v = QVBoxLayout(bar)
        v.setContentsMargins(14, 4, 14, 10)
        v.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        title = QLabel("SYSTEM  TERMINAL")
        title.setStyleSheet(
            f"color:{C.TXT_MUTED}; font-size:9px; font-weight:700;"
            f"letter-spacing:1.8px; background:transparent; border:none;")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(52, 22)
        clear_btn.setObjectName("ghost")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C.TXT_MUTED};
                border: 1px solid {C.BORDER};
                border-radius: 4px;
                font-size: 10px;
                padding: 2px 6px;
            }}
            QPushButton:hover {{ color: {C.TXT_SECONDARY}; }}
        """)
        clear_btn.clicked.connect(lambda: self._terminal.clear())
        header.addWidget(clear_btn)
        v.addLayout(header)

        self._terminal = TerminalLog()
        v.addWidget(self._terminal)
        return bar

    # ── Helper factory methods ─────────────────────────────────────────────────

    @staticmethod
    def _label(
        text: str, colour: str = C.TXT_PRIMARY,
        size: int = 13, bold: bool = False
    ) -> QLabel:
        lbl = QLabel(text)
        weight = "font-weight:700;" if bold else ""
        lbl.setStyleSheet(
            f"color:{colour}; font-size:{size}px; {weight}"
            f"background:transparent; border:none;")
        return lbl

    @staticmethod
    def _muted_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{C.TXT_SECONDARY}; font-size:11px;"
            f"background:transparent; border:none;")
        return lbl

    # ══════════════════════════════════════════════════════════════
    # SLOT / EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════

    def _log(self, msg: str, level: str = "INFO") -> None:
        """Route terminal messages from any thread safely through Qt signals."""
        self._terminal.append_line(msg, level)

    # ── Run analytics & backtest ───────────────────────────────────────────────

    def _on_run(self) -> None:
        ticker = self._ticker_input.text().strip().upper()
        if not ticker:
            self._log("Please enter a ticker symbol before running.", "WARN")
            return

        self._run_btn.setEnabled(False)
        self._progress.setValue(5)
        self._log(
            f"Analytics pipeline launched  ▸  {ticker}  "
            f"({self._asset_combo.currentText()})  |  VIX={self._vix:.1f}", "INFO")

        bt_worker = BacktestWorker(ticker, self._vix)
        bt_worker.result_ready.connect(self._on_backtest_done)
        bt_worker.log_msg.connect(self._log)
        bt_worker.progress.connect(self._progress.setValue)
        self._workers.append(bt_worker)
        bt_worker.start()

        # Kick off a simultaneous signal refresh
        self._on_refresh_signal()

    def _on_backtest_done(self, result: BacktestResult) -> None:
        self._run_btn.setEnabled(True)

        if result.error:
            self._log(f"Backtest failed: {result.error}", "ERROR")
            QTimer.singleShot(1500, lambda: self._progress.setValue(0))
            return

        # ── Update stats bar ───────────────────────────────────────────────────
        ret_colour = C.ACCENT_GREEN if result.total_return >= 0 else C.ACCENT_RED
        sign = "+" if result.total_return >= 0 else ""

        self._stat_return.set_value(
            f"{sign}{result.total_return:.2f}%", ret_colour)
        self._stat_winrate.set_value(
            f"{result.win_rate:.1f}%",
            C.ACCENT_GREEN if result.win_rate >= 50 else C.ACCENT_RED)
        self._stat_maxdd.set_value(
            f"{result.max_drawdown:.2f}%", C.ACCENT_RED)
        self._stat_sharpe.set_value(
            f"{result.sharpe:.2f}",
            C.ACCENT_GREEN if result.sharpe >= 1 else C.TXT_PRIMARY)
        self._stat_pf.set_value(
            f"{result.profit_factor:.2f}",
            C.ACCENT_GREEN if result.profit_factor >= 1 else C.ACCENT_RED)
        self._stat_trades.set_value(str(result.total_trades))
        self._stat_final.set_value(
            f"${result.final_balance:,.2f}", ret_colour)

        # ── Render chart ───────────────────────────────────────────────────────
        self._chart.plot_results(result)
        QTimer.singleShot(2000, lambda: self._progress.setValue(0))

    # ── Macro data refresh ─────────────────────────────────────────────────────

    def _fetch_macro(self) -> None:
        self._macro_btn.setEnabled(False)
        self._log("Global macro pipeline initiated…", "INFO")

        w = MacroDataWorker()
        w.data_ready.connect(self._on_macro_ready)
        w.log_msg.connect(self._log)
        w.finished.connect(lambda: self._macro_btn.setEnabled(True))
        self._workers.append(w)
        w.start()

    def _on_macro_ready(self, snap: Dict) -> None:
        if not snap:
            self._log("Macro data unavailable — using defaults.", "WARN")
            return

        self._macro_snap = snap
        self._vix        = snap.get("vix", 20.0)

        self._kpi_vix.set_value(f"{self._vix:.1f}")
        self._kpi_fed.set_value(f"{snap.get('fed_rate', 0):.2f}%")

        # ── VIX risk-level classification ──────────────────────────────────────
        if self._vix >= Config.VIX_FREEZE:
            self._kpi_risk_level.set_value("CRITICAL ⚠", C.ACCENT_RED)
            self._log(
                f"CRITICAL: VIX={self._vix:.1f} above freeze threshold "
                f"({Config.VIX_FREEZE}). Automated execution suspended.", "ERROR")
        elif self._vix >= Config.VIX_REDUCE:
            self._kpi_risk_level.set_value("ELEVATED ▲", C.ACCENT_AMBER)
            self._log(
                f"ELEVATED RISK: VIX={self._vix:.1f}  —  "
                f"Auto position sizes reduced 50%.", "WARN")
        else:
            self._kpi_risk_level.set_value("NORMAL ✓", C.ACCENT_GREEN)

    # ── Auto signal controls ───────────────────────────────────────────────────

    def _on_refresh_signal(self) -> None:
        ticker = self._ticker_input.text().strip().upper()
        if not ticker:
            return
        self._commit_btn.setEnabled(False)
        self._signal_badge.setText("LOADING…")
        self._signal_badge.setStyleSheet(
            f"color:{C.TXT_MUTED}; font-size:20px; font-weight:800;"
            f"background:transparent; border:none;")

        w = SignalWorker(ticker, self._balance, self._vix)
        w.signal_ready.connect(self._on_signal_ready)
        w.log_msg.connect(self._log)
        self._workers.append(w)
        w.start()

    def _on_signal_ready(
        self, sig: str, entry: float, sl: float, tp: float, size: float
    ) -> None:
        self._auto_data = {
            "signal": sig, "entry": entry,
            "sl": sl, "tp": tp, "size": size,
        }
        colour_map = {
            "BUY" : C.ACCENT_GREEN,
            "SELL": C.ACCENT_RED,
            "HOLD": C.ACCENT_AMBER,
        }
        clr = colour_map.get(sig, C.TXT_MUTED)
        self._signal_badge.setText(sig)
        self._signal_badge.setStyleSheet(
            f"color:{clr}; font-size:20px; font-weight:800;"
            f"background:transparent; border:none;")

        if entry > 0:
            atr_val = (sl - entry if sig == "SELL" else entry - sl) / Config.ATR_MULT_SL
            self._auto_entry.setText(f"${entry:.4f}")
            self._auto_sl   .setText(f"${sl:.4f}")
            self._auto_tp   .setText(f"${tp:.4f}")
            self._auto_size .setText(f"{size:.4f} units")
            self._auto_atr  .setText(f"{atr_val:.4f}")
            self._commit_btn.setEnabled(sig != "HOLD")
        else:
            for w in [self._auto_entry, self._auto_sl,
                      self._auto_tp, self._auto_size, self._auto_atr]:
                w.setText("—")

    def _on_commit_auto(self) -> None:
        d = self._auto_data
        if not d or d.get("signal") == "HOLD":
            return

        ticker = self._ticker_input.text().strip().upper()
        try:
            log_trade(
                trade_type="Auto",
                ticker=ticker,
                action=d["signal"],
                entry=d["entry"],
                size=d["size"],
                sl=d["sl"],
                tp=d["tp"],
                vix=self._vix,
            )
            self._log(
                f"Auto trade committed  ▸  {d['signal']} {ticker}  @  "
                f"${d['entry']:.4f}  |  SL ${d['sl']:.4f}  |  "
                f"TP ${d['tp']:.4f}  |  Size {d['size']:.4f} units",
                "TRADE",
            )
            self._log(f"CSV updated  ▸  {Config.CSV_LOG_PATH}", "SUCCESS")
            self._commit_btn.setEnabled(False)
        except Exception as exc:
            self._log(f"Failed to commit auto trade: {exc}", "ERROR")

    # ── Manual trade builder ───────────────────────────────────────────────────

    def _update_size_preview(self) -> None:
        """Recalculate and display position size as the user types."""
        try:
            entry = float(self._m_entry.text())
            sl    = float(self._m_sl.text())
            dist  = abs(entry - sl)
            if dist > 0:
                size      = (self._balance * Config.RISK_PER_TRADE) / dist
                risk_usd  = self._balance * Config.RISK_PER_TRADE
                self._m_size_preview.setText(
                    f"Position size: {size:.4f} units  "
                    f"(${risk_usd:.2f} at risk  =  1% of ${self._balance:,.2f})")
            else:
                self._m_size_preview.setText("Position size: —")
        except (ValueError, ZeroDivisionError):
            self._m_size_preview.setText("Position size: —")

    def _on_manual_execute(self) -> None:
        ticker = self._m_ticker.text().strip().upper()
        action = self._m_action.currentText()

        try:
            entry = float(self._m_entry.text())
            sl    = float(self._m_sl.text())
            tp    = float(self._m_tp.text())
        except ValueError:
            self._log(
                "Manual trade: all price fields must be valid numbers.", "WARN")
            return

        if not ticker:
            self._log("Manual trade: ticker symbol cannot be blank.", "WARN")
            return

        self._log(
            f"Validating manual {action} {ticker}  @  ${entry:.4f}  |  "
            f"SL ${sl:.4f}  |  TP ${tp:.4f}…", "INFO")

        w = ManualTradeWorker(ticker, action, entry, sl, tp,
                              self._balance, self._vix)
        w.trade_done.connect(self._on_manual_done)
        w.log_msg.connect(self._log)
        self._workers.append(w)
        w.start()

    def _on_manual_done(self, success: bool, message: str, size: float) -> None:
        if not success:
            self._log(f"Trade rejected: {message}", "WARN")

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Gracefully stop all background threads before the window closes."""
        for w in self._workers:
            try:
                if w.isRunning():
                    w.quit()
                    w.wait(800)
            except Exception:
                pass
        event.accept()


# ==============================================================================
# SECTION 6 — APPLICATION ENTRY POINT
# ==============================================================================

def main() -> None:
    """
    Bootstrap the macOS application with Retina / high-DPI support,
    a Fusion palette overridden for full dark-mode compliance, and
    the custom stylesheet applied globally.
    """
    # High-DPI / Retina display pass-through (must be set before QApplication)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.VERSION)
    app.setOrganizationName("QuantDev")
    app.setOrganizationDomain("quantdev.terminal")

    # ── Fusion style gives consistent cross-Mac rendering ──────────────────────
    app.setStyle("Fusion")

    # ── Dark QPalette (fallback for any widgets not covered by the stylesheet) ─
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(C.BG_DARK))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(C.TXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Base,            QColor(C.BG_MED))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(C.BG_CARD))
    pal.setColor(QPalette.ColorRole.Text,            QColor(C.TXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Button,          QColor(C.BG_MED))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(C.TXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(C.ACCENT_BLUE))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.Link,            QColor(C.ACCENT_BLUE))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(C.BG_CARD))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(C.TXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(C.TXT_MUTED))
    app.setPalette(pal)

    # ── Apply the custom dark stylesheet ──────────────────────────────────────
    app.setStyleSheet(DARK_QSS)

    # ── Create and display the main window ────────────────────────────────────
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
