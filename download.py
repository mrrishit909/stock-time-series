"""Step 1: daily adjusted prices of 17 US stocks and ETFs, plus two FRED series (S&P 500 index, VIX).

    ./venv/bin/python download.py   -> data/bars.csv (Alpaca), data/SP500.csv VIXCLS.csv (FRED), results/pull.json

Alpaca Markets market-data API (https://data.alpaca.markets/v2/stocks/bars, timeframe 1Day, feed=sip, adjustment=all: split- and dividend-adjusted closes, from 2016-01-04), read with a free paper-trading
account; the keys are read from a .env file (ALPACA_API_KEY, ALPACA_SECRET_KEY) inside this script and never printed or stored. Alpaca's market-data terms forbid redistributing the raw data, so
data/ is never committed and the repo holds only our own aggregates and statistics. feed=sip is the consolidated tape (all US exchanges); the free-tier iex feed starts only in July 2020, so it is not used.
FRED (Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/series/<id>): SP500 (S&P Dow Jones Indices; price index, 10 years of history; FRED marks it "copyrighted: pre-approval required") and
VIXCLS (CBOE; "copyrighted: citation required"); fetched with the default User-Agent (FRED stalls on custom ones). Only statistics computed from them (a correlation, a slope) are published. Set ENV_FILE to point at another .env.
"""
import csv
import hashlib
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import certifi

HERE = Path(__file__).parent
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM", "JNJ", "WMT", "KO", "SPY",
           "MCD", "SBUX", "YUM", "CMG", "DPZ", "QSR", "WEN", "DRI"]
START = "2016-01-01"
FEED = "sip"          # consolidated tape; the free plan serves it for bars older than 15 minutes. The iex feed only goes back to 2020-07-27 and would miss the 2020 crash.
FRED = ["SP500", "VIXCLS"]
CTX = ssl.create_default_context(cafile=certifi.where())


def keys():
    env = {}
    for line in open(os.environ.get("ENV_FILE", os.path.expanduser("~/trading-robot/.env"))):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip("'\"")
    return {"APCA-API-KEY-ID": env["ALPACA_API_KEY"], "APCA-API-SECRET-KEY": env["ALPACA_SECRET_KEY"]}


def get(url, headers=None):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), context=CTX, timeout=60).read()
        except OSError as e:
            if attempt == 4 or getattr(e, "code", 0) in (401, 403, 404, 422): raise
            time.sleep(5)


def main():
    (HERE / "data").mkdir(exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
    hdr = keys(); rows = []; end = (date.today() - timedelta(days=1)).isoformat()          # the free plan refuses sip bars from the last 15 minutes
    for sym in TICKERS:
        tok = None
        while True:
            q = {"symbols": sym, "timeframe": "1Day", "start": START, "end": end, "adjustment": "all", "feed": FEED, "limit": 10000, "sort": "asc"}
            if tok: q["page_token"] = tok
            j = json.loads(get("https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q), hdr))
            rows += [(sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"]) for b in j["bars"].get(sym, [])]
            tok = j.get("next_page_token")
            if not tok: break
    with open(HERE / "data" / "bars.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["symbol", "date", "open", "high", "low", "close", "volume"]); w.writerows(rows)
    raw = (HERE / "data" / "bars.csv").read_bytes()
    info = {"pulled": date.today().isoformat(), "files": {"bars.csv": {"url": "https://data.alpaca.markets/v2/stocks/bars (feed=sip, adjustment=all, 1Day, symbols: " + " ".join(TICKERS) + ")", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}}
    for s in FRED:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"; raw = get(url)          # default User-Agent: FRED stalls on custom ones
        (HERE / "data" / f"{s}.csv").write_bytes(raw)
        info["files"][f"{s}.csv"] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v["bytes"] for k, v in info["files"].items()}), len(rows), "bars")


if __name__ == "__main__":
    main()
