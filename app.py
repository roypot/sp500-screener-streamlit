# app.py
import os, time, math, requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pandas_ta as ta

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="S&P 500 Stock Rater", page_icon="📈", layout="wide")
st.title("S&P 500 Stock Rater (100-point model)")

# ------------------------
# Settings & constants
# ------------------------
FMP_KEY = st.secrets.get("FMP_API_KEY", "")
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SESSION = requests.Session()

# Indicator settings
RSI_LEN = 14
ADX_LEN = 14
ROC_LEN = 252   # ~12m daily
SMA_LEN = 200

# ------------------------
# Helpers
# ------------------------
@st.cache_data(ttl=6*60*60)
def load_sp500_tickers() -> pd.DataFrame:
    # Wikipedia scrape
    tables = pd.read_html(WIKI_SP500_URL)  # table 0 usually the constituents
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "company", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B for Yahoo
    return df[["ticker", "company", "sector"]]

@st.cache_data(ttl=24*60*60, show_spinner=False)
def fmp_ratios(symbols: list[str]) -> pd.DataFrame:
    """
    Pulls key ratios & metrics we need from FMP Stable API.
    We call 2 endpoints and merge:
      - /stable/ratios?symbol=...
      - /stable/key-metrics?symbol=...
    """
    if not FMP_KEY:
        return pd.DataFrame()

    def fetch(endpoint, params):
        url = f"https://financialmodelingprep.com/stable/{endpoint}"
        params = params | {"apikey": FMP_KEY}
        r = SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    rows = []
    for sym in symbols:
        # Latest "FY" is fine for annual comparisons
        try:
            ratios = fetch("ratios", {"symbol": sym, "limit": 1, "period": "FY"})
            km = fetch("key-metrics", {"symbol": sym, "limit": 1, "period": "FY"})
            r = ratios[0] if ratios else {}
            k = km[0] if km else {}

            rows.append({
                "ticker": sym,
                # Valuation
                "ps_ratio": r.get("priceToSalesRatio", np.nan) or r.get("priceToSalesRatioTTM", np.nan),
                "peg_ratio": r.get("pegRatio", np.nan) or r.get("pegRatioTTM", np.nan),
                "fcf_yield": k.get("freeCashFlowYield", np.nan) or k.get("freeCashFlowYieldTTM", np.nan),
                # Quality
                "de_ratio": r.get("debtToEquity", np.nan) or r.get("debtToEquityTTM", np.nan),
                "gross_margin": r.get("grossProfitMargin", np.nan) or r.get("grossProfitMarginTTM", np.nan),
                "revenue_growth": r.get("revenueGrowth", np.nan) or r.get("revenueGrowthTTM", np.nan),
            })
        except Exception:
            # Be resilient to intermittent API failures
            rows.append({"ticker": sym})

        time.sleep(0.12)  # polite pacing for free tier
    return pd.DataFrame(rows)

@st.cache_data(ttl=12*60*60, show_spinner=False)
def price_history(symbol: str, period="3y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return df
    df["SMA200"] = ta.sma(df["Close"], length=SMA_LEN)
    df["RSI"] = ta.rsi(df["Close"], length=RSI_LEN)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
    df["ADX"] = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=ADX_LEN)["ADX_14"]
    df["ROC"] = ta.roc(df["Close"], length=ROC_LEN)
    return df.dropna().copy()

def pctile(series, x):
    s = pd.Series(series).dropna()
    if len(s) == 0 or pd.isna(x):
        return np.nan
    return (s.rank(pct=True, method="max")[s.index.get_indexer([s.index[s.sub(x).abs().idxmin()]])[0]] 
            if x in s.values else (s.lt(x).mean()))

def score_low_good(x, peer):
    p = pctile(peer, x)
    return 10 * (1 - p) if pd.notna(p) else np.nan

def score_high_good(x, peer):
    p = pctile(peer, x)
    return 10 * p if pd.notna(p) else np.nan

def score_rsi(rsi):
    # 10 at 50; 0 at ≤30 or ≥70
    if pd.isna(rsi):
        return np.nan
    return max(0.0, 10 * (1 - abs(rsi - 50) / 20))

def score_macd(macd, signal):
    if pd.isna(macd) or pd.isna(signal):
        return np.nan
    if macd > 0 and macd > signal:
        return 10.0
    if macd > signal:
        return 6.0
    return 0.0

def score_price_vs_200(close, sma200):
    if pd.isna(close) or pd.isna(sma200) or sma200 == 0:
        return np.nan
    prem = (close / sma200) - 1
    return 5.0 * float(np.clip(prem / 0.10, 0, 1))

def score_adx(adx):
    if pd.isna(adx):
        return np.nan
    if adx >= 25: return 10.0
    if adx >= 20: return 7.0
    if adx >= 15: return 3.0
    return 0.0

def score_roc(roc, peer):
    p = pctile(peer, roc)
    return 10 * p if pd.notna(p) else np.nan

# ------------------------
# UI controls
# ------------------------
meta = load_sp500_tickers()
left, right = st.columns([1,3])
with left:
    st.caption(f"Universe size: **{len(meta)}** (live from Wikipedia)")
    pick_sector = st.selectbox("Filter sector (optional)", ["All"] + sorted(meta["sector"].unique().tolist()))
    top_n = st.slider("Show Top N", 10, 200, 50, step=10)
    run_btn = st.button("Run Screening", type="primary")
with right:
    st.info("Full 100‑point scoring requires an **FMP API key** in secrets. "
            "Without it, the app will still compute **Technicals & Trend** (45 pts).")

# ------------------------
# Run
# ------------------------
if run_btn:
    universe = meta.copy()
    if pick_sector != "All":
        universe = universe[universe["sector"] == pick_sector]

    symbols = universe["ticker"].tolist()

    # Fundamentals (if key present)
    fund = fmp_ratios(symbols)
    have_fund = not fund.empty

    # Technicals: compute only the latest row for each symbol
    tech_rows = []
    for sym in symbols:
        df = price_history(sym)
        if df.empty: 
            continue
        last = df.iloc[-1]
        tech_rows.append({
            "ticker": sym,
            "close": last["Close"],
            "rsi": last.get("RSI"),
            "macd": last.get("MACD"),
            "macd_signal": last.get("MACD_SIGNAL"),
            "sma200": last.get("SMA200"),
            "adx": last.get("ADX"),
            "roc": last.get("ROC"),
        })
    tech = pd.DataFrame(tech_rows)

    # Merge
    df = universe.merge(tech, on="ticker", how="left")
    if have_fund:
        df = df.merge(fund, on="ticker", how="left")

    # Peer vectors for percentile scoring
    peers = {}
    if have_fund:
        peers["ps_ratio"] = df["ps_ratio"]
        peers["peg_ratio"] = df["peg_ratio"]
        peers["fcf_yield"] = df["fcf_yield"]
        peers["de_ratio"] = df["de_ratio"]
        peers["gross_margin"] = df["gross_margin"]
        peers["revenue_growth"] = df["revenue_growth"]
    peers["roc"] = df["roc"]

    # Compute scores
    if have_fund:
        df["score_ps"]   = df["ps_ratio"].apply(lambda x: score_low_good(x, peers["ps_ratio"]))
        df["score_peg"]  = df["peg_ratio"].apply(lambda x: score_low_good(x, peers["peg_ratio"]))
        df["score_fcfy"] = df["fcf_yield"].apply(lambda x: score_high_good(x, peers["fcf_yield"]))
        df["score_de"]   = df["de_ratio"].apply(lambda x: score_low_good(x, peers["de_ratio"]))
        df["score_gm"]   = df["gross_margin"].apply(lambda x: score_high_good(x, peers["gross_margin"]))
        df["score_rev"]  = df["revenue_growth"].apply(lambda x: 0.5 * score_high_good(x, peers["revenue_growth"]))  # 5 max (0.5*10)
    else:
        for c in ["score_ps","score_peg","score_fcfy","score_de","score_gm","score_rev"]:
            df[c] = np.nan

    df["score_rsi"]  = df["rsi"].apply(score_rsi)
    df["score_macd"] = df.apply(lambda r: score_macd(r["macd"], r["macd_signal"]), axis=1)
    df["score_pv200"]= df.apply(lambda r: score_price_vs_200(r["close"], r["sma200"]), axis=1)
    df["score_adx"]  = df["adx"].apply(score_adx)
    df["score_roc"]  = df["roc"].apply(lambda x: score_roc(x, peers["roc"]))

    # Category totals
    df["Valuation"] = df[["score_ps","score_peg","score_fcfy"]].sum(axis=1, min_count=1)
    df["Quality"]   = df[["score_de","score_gm","score_rev"]].sum(axis=1, min_count=1)
    df["Momentum"]  = df[["score_rsi","score_macd","score_pv200"]].sum(axis=1, min_count=1)
    df["Trend"]     = df[["score_adx","score_roc"]].sum(axis=1, min_count=1)

    # Total (handle missing fundamentals gracefully)
    df["TotalScore"] = df[["Valuation","Quality","Momentum","Trend"]].sum(axis=1, min_count=1)

    # Sort & show
    out_cols = ["ticker","company","sector","TotalScore","Valuation","Quality","Momentum","Trend",
                "ps_ratio","peg_ratio","fcf_yield","de_ratio","gross_margin","revenue_growth",
                "rsi","macd","macd_signal","adx","roc","close"]
    view = df[out_cols].sort_values("TotalScore", ascending=False).head(top_n).reset_index(drop=True)
    st.subheader("Top Ranked")
    st.dataframe(view, use_container_width=True)

    # Download
    st.download_button("Download CSV", view.to_csv(index=False).encode(), file_name="sp500_ratings.csv", mime="text/csv")

    # Drill‑down charts
    sel = st.selectbox("Show charts for:", view["ticker"])
    if sel:
        hist = price_history(sel, period="3y")
        if not hist.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{sel} – Price with 200‑SMA**")
                st.line_chart(hist[["Close","SMA200"]])
                st.caption("Prices via Yahoo Finance (`yfinance`). Indicators via `pandas_ta`.")
            with c2:
                st.markdown("**RSI & MACD**")
                st.line_chart(hist["RSI"])
                macd_plot = hist[["MACD","MACD_SIGNAL"]].dropna()
                if not macd_plot.empty:
                    st.line_chart(macd_plot)
            c3, _ = st.columns([1,1])
            with c3:
                st.markdown("**ADX & ROC**")
                st.line_chart(hist[["ADX","ROC"]])
