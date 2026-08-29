#!/usr/bin/env python3
"""
Scan a universe of stocks for:
    RSI(14) > 50           (optionally with a recent regular bullish divergence)
    AND MACD2 short histogram in its "dark blue" state (#42a5f5):
        hist2 >= 0 and hist2[2] < hist2

Runs the check on 15m, 1h and 1d candles and pushes the result list to Telegram.

Setup:
    pip install yfinance pandas numpy requests
    export TELEGRAM_BOT_TOKEN="123456:ABC..."
    export TELEGRAM_CHAT_ID="-1001234567890"
    python scanner.py --symbols-file nse500.txt --timeframes 15m 1h 1d

Add --require-div to demand a bullish RSI divergence within the last N bars.
Add --loop 15 to keep it running and re-scan every 15 minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None

load_dotenv()


# ──────────────────────────────── indicators ────────────────────────────────

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI, matching Pine's ta.rsi()."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(roll_down.notna(), np.nan)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def pivot_flags(values: np.ndarray, lbL: int, lbR: int, low_pivot: bool) -> np.ndarray:
    """
    Mirror of Pine's ta.pivotlow / ta.pivothigh.
    Returns a boolean array; index i is True when values[i - lbR] is the pivot,
    i.e. the pivot is *confirmed* at bar i.
    """
    n = len(values)
    found = np.zeros(n, dtype=bool)
    for i in range(lbL + lbR, n):
        c = i - lbR
        window = values[c - lbL: c + lbR + 1]
        if np.isnan(window).any():
            continue
        pivot = values[c]
        left, right = window[:lbL], window[lbL + 1:]
        if low_pivot:
            if (left > pivot).all() and (right > pivot).all():
                found[i] = True
        else:
            if (left < pivot).all() and (right < pivot).all():
                found[i] = True
    return found


def bullish_divergence(osc: np.ndarray, low: np.ndarray, lbL: int, lbR: int,
                       range_lower: int, range_upper: int) -> np.ndarray:
    """Regular bullish divergence: RSI higher low while price makes a lower low."""
    found = pivot_flags(osc, lbL, lbR, low_pivot=True)
    idx = np.flatnonzero(found)
    sig = np.zeros(len(osc), dtype=bool)
    for prev, cur in zip(idx, idx[1:]):
        bars = cur - prev - 1                      # Pine: barssince(plFound[1])
        if not (range_lower <= bars <= range_upper):
            continue
        if osc[cur - lbR] > osc[prev - lbR] and low[cur - lbR] < low[prev - lbR]:
            sig[cur] = True
    return sig


def bearish_divergence(osc: np.ndarray, high: np.ndarray, lbL: int, lbR: int,
                       range_lower: int, range_upper: int) -> np.ndarray:
    found = pivot_flags(osc, lbL, lbR, low_pivot=False)
    idx = np.flatnonzero(found)
    sig = np.zeros(len(osc), dtype=bool)
    for prev, cur in zip(idx, idx[1:]):
        bars = cur - prev - 1
        if not (range_lower <= bars <= range_upper):
            continue
        if osc[cur - lbR] < osc[prev - lbR] and high[cur - lbR] > high[prev - lbR]:
            sig[cur] = True
    return sig


# ──────────────────────────────── evaluation ────────────────────────────────

@dataclass
class Hit:
    symbol: str
    timeframe: str
    close: float
    rsi: float
    hist2: float
    div_bars_ago: int | None


@dataclass
class Params:
    rsi_len: int = 14
    lbL: int = 5
    lbR: int = 5
    range_lower: int = 5
    range_upper: int = 60
    fast2: int = 34
    slow2: int = 144
    signal2: int = 6
    require_div: bool = False
    div_lookback: int = 10


def evaluate(df: pd.DataFrame, symbol: str, tf: str, p: Params,
             drop_forming: bool) -> Hit | None:
    """Return a Hit if the last completed bar satisfies both conditions."""
    df = df.dropna(subset=["Close"])
    if drop_forming and len(df) > 1:
        df = df.iloc[:-1]                          # ignore the still-forming candle
    if len(df) < max(p.slow2 + p.signal2, p.rsi_len, p.lbL + p.lbR) + 10:
        return None

    close = df["Close"].astype(float)
    osc = rsi(close, p.rsi_len)

    macd2 = ema(close, p.fast2) - ema(close, p.slow2)
    signal2 = ema(macd2, p.signal2)
    hist2 = (macd2 - signal2).to_numpy()

    o = osc.to_numpy()
    if np.isnan(o[-1]) or np.isnan(hist2[-1]) or np.isnan(hist2[-3]):
        return None

    rsi_ok = o[-1] > 50.0
    # #42a5f5 == col_grow_above2: histogram above zero and rising vs 2 bars back
    macd_blue = hist2[-1] >= 0.0 and hist2[-3] < hist2[-1]
    if not (rsi_ok and macd_blue):
        return None

    div_bars_ago = None
    if p.require_div or True:                       # always report it when present
        sig = bullish_divergence(o, df["Low"].astype(float).to_numpy(),
                                 p.lbL, p.lbR, p.range_lower, p.range_upper)
        where = np.flatnonzero(sig)
        if len(where):
            div_bars_ago = int(len(sig) - 1 - where[-1])

    if p.require_div and (div_bars_ago is None or div_bars_ago > p.div_lookback):
        return None

    return Hit(symbol, tf, float(close.iloc[-1]), float(o[-1]),
               float(hist2[-1]), div_bars_ago)


# ──────────────────────────────── data ────────────────────────────────

# yfinance limits: 15m data goes back ~60 days, 1h ~730 days.
PERIOD_FOR = {"15m": "60d", "30m": "60d", "1h": "180d", "60m": "180d",
              "1d": "2y", "1wk": "5y"}
YF_INTERVAL = {"1h": "60m"}


def fetch_batch(symbols: list[str], tf: str) -> dict[str, pd.DataFrame]:
    if yf is None:
        sys.exit("yfinance is not installed. Run: pip install yfinance")
    interval = YF_INTERVAL.get(tf, tf)
    period = PERIOD_FOR.get(tf, "1y")
    data = yf.download(symbols, period=period, interval=interval,
                       group_by="ticker", auto_adjust=False, threads=True,
                       progress=False)
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            d = data[s] if len(symbols) > 1 else data
            if isinstance(d, pd.DataFrame) and not d.empty:
                out[s] = d
        except (KeyError, TypeError):
            continue
    return out


# ──────────────────────────────── telegram ────────────────────────────────

def send_telegram(text: str, token: str, chat_id: str) -> None:
    if requests is None:
        sys.exit("requests is not installed. Run: pip install requests")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk in [text[i:i + 3800] for i in range(0, len(text), 3800)] or [text]:
        r = requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                     "parse_mode": "HTML",
                                     "disable_web_page_preview": True}, timeout=30)
        if not r.ok:
            print(f"Telegram error {r.status_code}: {r.text}", file=sys.stderr)


def format_report(hits: list[Hit], timeframes: list[str]) -> str:
    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    lines = [f"<b>RSI &gt; 50 + MACD2 dark blue</b>  ({stamp})"]
    for tf in timeframes:
        rows = [h for h in hits if h.timeframe == tf]
        lines.append(f"\n<b>── {tf} ── {len(rows)} stock(s)</b>")
        if not rows:
            lines.append("none")
            continue
        for h in sorted(rows, key=lambda x: -x.rsi):
            tag = f"  div {h.div_bars_ago}b ago" if h.div_bars_ago is not None and h.div_bars_ago <= 20 else ""
            lines.append(f"{h.symbol}  ₹{h.close:,.2f}  RSI {h.rsi:.1f}{tag}")
    return "\n".join(lines)


# ──────────────────────────────── universe ────────────────────────────────

DEFAULT_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "HCLTECH.NS",
    "TECHM.NS", "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "SHREECEM.NS",
    "INDUSINDBK.NS", "HINDALCO.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "BPCL.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "UPL.NS", "BAJAJFINSV.NS", "M&M.NS",
]


def load_symbols(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_UNIVERSE
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


# ──────────────────────────────── main ────────────────────────────────

def scan(symbols: list[str], timeframes: list[str], p: Params,
         batch_size: int, drop_forming: bool) -> list[Hit]:
    hits: list[Hit] = []
    for tf in timeframes:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            frames = fetch_batch(batch, tf)
            for sym, df in frames.items():
                try:
                    hit = evaluate(df, sym, tf, p, drop_forming)
                except Exception as e:                       # noqa: BLE001
                    print(f"{sym} {tf}: {e}", file=sys.stderr)
                    continue
                if hit:
                    hits.append(hit)
            time.sleep(1)                                    # be polite to the API
        print(f"{tf}: scanned {len(symbols)} symbols, "
              f"{len([h for h in hits if h.timeframe == tf])} hits", file=sys.stderr)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols-file", help="one ticker per line, e.g. RELIANCE.NS")
    ap.add_argument("--timeframes", nargs="+", default=["15m", "1h", "1d"])
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--require-div", action="store_true",
                    help="also require a recent regular bullish RSI divergence")
    ap.add_argument("--div-lookback", type=int, default=10)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--include-forming", action="store_true",
                    help="evaluate the still-forming candle instead of the last closed one")
    ap.add_argument("--loop", type=int, default=0, help="re-scan every N minutes")
    ap.add_argument("--print-only", action="store_true", help="skip Telegram")
    args = ap.parse_args()

    p = Params(rsi_len=args.rsi_len, require_div=args.require_div,
               div_lookback=args.div_lookback)
    symbols = load_symbols(args.symbols_file)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not args.print_only and not (token and chat_id):
        sys.exit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or pass --print-only")

    while True:
        hits = scan(symbols, args.timeframes, p, args.batch_size,
                    drop_forming=not args.include_forming)
        report = format_report(hits, args.timeframes)
        if args.print_only:
            print(report)
        else:
            send_telegram(report, token, chat_id)
        if not args.loop:
            break
        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
