"""
Shiomega Daily — Streamlit dashboard.  潮目が変わる — "the tide has turned."

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub -> share.streamlit.io -> pick app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import shiomega_core as sc
import shiomega_engine as se

try:
    import yfinance as yf
    YF = True
except Exception:                                 # noqa: BLE001
    YF = False

st.set_page_config(page_title="Shiomega Daily · 潮目が変わる",
                   page_icon="🌊", layout="wide")

# ----------------------------- palette ------------------------------------- #
INK, PANEL, PANEL2, LINE = "#0A111E", "#101A2B", "#15223A", "#22334F"
TEXT, MUT, FAINT = "#E9EEF6", "#8CA0BC", "#5C7089"
GOLD, TENKAN, KIJUN = "#E2B23F", "#4EA3E8", "#E15B5B"
POS, NEG, WATCH = "#3FB77E", "#E06060", "#5FA8D9"
GREEN_CLOUD, RED_CLOUD = "rgba(63,183,126,0.13)", "rgba(224,96,96,0.12)"
CHIKOU = "#9FD08A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Shippori+Mincho:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {{ background:{INK}; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.2rem; max-width:1180px; }}
h1,h2,h3 {{ color:{TEXT}; }}
.sh-head {{ display:flex; align-items:flex-end; gap:20px; flex-wrap:wrap;
  border-bottom:1px solid {LINE}; padding-bottom:14px; margin-bottom:6px; }}
.sh-kanji {{ font-family:'Shippori Mincho',serif; font-size:52px; line-height:1;
  color:{GOLD}; font-weight:700; }}
.sh-title {{ font-family:'Shippori Mincho',serif; font-size:30px; font-weight:700;
  letter-spacing:.06em; color:{TEXT}; }}
.sh-sub {{ color:{MUT}; font-size:13.5px; margin-top:3px; }}
.sh-sub b {{ color:{TEXT}; }}
.sh-badge {{ margin-left:auto; align-self:center; font-family:'IBM Plex Mono',monospace;
  font-size:12px; letter-spacing:.14em; color:{GOLD}; border:1px solid #8A6E2B;
  padding:6px 12px; border-radius:3px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color:{MUT}; font-weight:500; padding:8px 16px; }}
.stTabs [aria-selected="true"] {{ color:{GOLD}; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE};
  border-radius:6px; padding:12px 14px; }}
[data-testid="stMetricLabel"] {{ color:{FAINT}; font-size:11px !important;
  letter-spacing:.06em; text-transform:uppercase; }}
[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace;
  font-size:20px !important; color:{TEXT}; }}
.stDataFrame {{ border:1px solid {LINE}; border-radius:6px; }}
.step-card {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:6px;
  padding:13px 15px; height:100%; }}
.step-card .n {{ font-family:'Shippori Mincho',serif; color:{GOLD};
  font-size:20px; font-weight:700; }}
.step-card h4 {{ color:{TEXT}; font-size:14px; margin:5px 0 4px; }}
.step-card p {{ color:{MUT}; font-size:12.5px; margin:0; }}
.gate-tbl {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
.gate-tbl td {{ padding:9px 12px; border-bottom:1px solid #1A283F; vertical-align:top; color:{TEXT}; }}
.gate-tbl td:first-child {{ font-family:'IBM Plex Mono',monospace; color:{GOLD};
  white-space:nowrap; width:52px; }}
.note {{ color:{FAINT}; font-size:12px; }}
.mono {{ font-family:'IBM Plex Mono',monospace; color:{GOLD}; }}
</style>

<div class="sh-head">
  <div class="sh-kanji">潮目</div>
  <div>
    <div class="sh-title">SHIOMEGA</div>
    <div class="sh-sub"><b>The tide has turned.</b> Tenkan/Kijun cross is the heads-up ·
      the clean Chikou breakout is the trigger.</div>
  </div>
  <div class="sh-badge">DAILY · 9 / 26 / 52</div>
</div>
""", unsafe_allow_html=True)

GRID = "rgba(24,37,64,0.5)"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=MUT, family="IBM Plex Mono"),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False),
    showlegend=False, hovermode="x unified",
)


# =========================================================================== #
# Ichimoku chart (Plotly) with the four annotated moments
# =========================================================================== #
def ichimoku_fig(df: pd.DataFrame, setup: dict, side: str, title: str = "") -> go.Figure:
    d = setup["df"]
    if d is None or len(d) < 2:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", showarrow=False,
                           font=dict(color=MUT, size=14))
        fig.update_layout(**PLOT_LAYOUT, height=460)
        return fig
    c, b = setup["cross"], setup["trigger"]
    n = len(d)
    if c is not None:
        i0, i1 = max(0, c - 58), min(n - 1, b + 42)
    else:
        i0, i1 = max(0, n - 160), n - 1
    idx = d.index
    fig = go.Figure()

    # Kumo (displaced +26): plot cloud_a / cloud_b already shifted in core
    ca, cb = d["cloud_a"], d["cloud_b"]
    fig.add_trace(go.Scatter(x=idx, y=ca, line=dict(color="rgba(63,183,126,.55)", width=1),
                             name="Span A", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=cb, fill="tonexty",
                             fillcolor=GREEN_CLOUD, line=dict(color="rgba(224,96,96,.5)", width=1),
                             name="Span B", hoverinfo="skip"))

    fig.add_trace(go.Candlestick(
        x=idx, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        increasing=dict(line=dict(color="#57907A"), fillcolor="#2E8663"),
        decreasing=dict(line=dict(color="#9A5560"), fillcolor="#B14F58"),
        name="Price", showlegend=False))

    fig.add_trace(go.Scatter(x=idx, y=d["tenkan"], line=dict(color=TENKAN, width=1.6),
                             name="Tenkan"))
    fig.add_trace(go.Scatter(x=idx, y=d["kijun"], line=dict(color=KIJUN, width=1.6),
                             name="Kijun"))
    # Chikou: close shifted 26 bars back
    chikou = d["Close"].shift(-sc.DISP)
    fig.add_trace(go.Scatter(x=idx, y=chikou, line=dict(color=CHIKOU, width=1.2, dash="dot"),
                             name="Chikou", opacity=.85))

    if c is not None and b is not None:
        # ② wait band
        fig.add_vrect(x0=idx[c], x1=idx[b], fillcolor="rgba(140,160,188,.06)",
                      line_width=0, layer="below")
        # past-price ceiling/floor the Chikou had to clear
        win_lo, win_hi = max(0, b - 31), b - 21
        if side == "long":
            lvl = float(d["High"].iloc[win_lo:win_hi + 1].max())
        else:
            lvl = float(d["Low"].iloc[win_lo:win_hi + 1].min())
        fig.add_shape(type="line", x0=idx[max(i0, win_lo)], x1=idx[b], y0=lvl, y1=lvl,
                      line=dict(color=FAINT, width=1, dash="dot"))
        # ① cross marker
        fig.add_trace(go.Scatter(x=[idx[c]], y=[d["tenkan"].iloc[c]], mode="markers",
                                 marker=dict(color="rgba(0,0,0,0)", size=16,
                                             line=dict(color=TENKAN, width=2.4)),
                                 name="cross", hoverinfo="skip"))
        fig.add_annotation(x=idx[c], y=d["tenkan"].iloc[c], text="① TK cross",
                           showarrow=True, arrowcolor=TENKAN, arrowhead=0,
                           ay=(34 if side == "long" else -30), ax=0,
                           font=dict(color=TEXT, size=12))
        # ③ entry + stop
        fig.add_annotation(x=idx[b], y=d["Close"].iloc[b], text="③ Chikou breakout — enter",
                           showarrow=True, arrowcolor=GOLD, arrowhead=2, arrowwidth=2,
                           ay=(60 if side == "long" else -50), ax=0,
                           font=dict(color=GOLD, size=12))
        fig.add_shape(type="line", x0=idx[max(0, b - 6)],
                      x1=idx[min(n - 1, b + 16)], y0=setup["stop"], y1=setup["stop"],
                      line=dict(color=NEG, width=1.4, dash="dash"))
        fig.add_annotation(x=idx[min(n - 1, b + 16)], y=setup["stop"],
                           text="stop · Kijun / swing", showarrow=False,
                           font=dict(color=NEG, size=11), xanchor="right",
                           yshift=(-10 if side == "long" else 10))

    fig.update_layout(**PLOT_LAYOUT, height=460,
                      xaxis_rangeslider_visible=False,
                      xaxis_range=[idx[i0], idx[i1]])
    return fig


# =========================================================================== #
# Tabs
# =========================================================================== #
tab_setup, tab_sig, tab_bt, tab_rules = st.tabs(
    ["Setup", "Signals board", "Backtest analyzer", "Rules & parameters"])

# --------------------------------------------------------------------------- #
# SETUP
# --------------------------------------------------------------------------- #
with tab_setup:
    st.markdown("#### The four moments of the trade")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 3])
    with c1:
        side = st.radio("Side", ["long", "short"], horizontal=True,
                        format_func=str.upper, label_visibility="collapsed")
    with c2:
        mode = st.radio("Source", ["Synthetic", "Live ticker"], horizontal=True,
                        label_visibility="collapsed",
                        disabled=not YF, help=None if YF else "yfinance not installed")
    live_ticker = ""
    with c3:
        if mode == "Live ticker" and YF:
            live_ticker = st.text_input("Ticker", value="AAPL",
                                        label_visibility="collapsed",
                                        placeholder="e.g. AAPL, SHOP.TO")
    with c4:
        reroll = st.button("↻ New synthetic tape") if mode == "Synthetic" else False

    if "seed" not in st.session_state:
        st.session_state.seed = 7
    if reroll:
        st.session_state.seed = int(np.random.randint(1, 1_000_000))

    df = None
    if mode == "Live ticker" and YF and live_ticker.strip():
        try:
            raw = yf.download(live_ticker.strip(), period="2y", interval="1d",
                              auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            cand = raw[["Open", "High", "Low", "Close"]].dropna()
            if len(cand) >= 90:
                df = cand
            else:
                st.warning(f"Not enough daily history for {live_ticker}; showing a synthetic tape.")
        except Exception as e:                    # noqa: BLE001
            st.warning(f"Could not load {live_ticker}: {e}. Showing a synthetic tape.")
    if df is None:
        seed = st.session_state.seed
        # reroll until the synthetic tape actually contains a setup
        for _ in range(50):
            df = sc.gen_series(side, seed)
            if sc.find_setup(df, side)["trigger"] is not None:
                break
            seed += 1
        st.session_state.seed = seed

    setup = sc.find_setup(df, side)
    if setup["trigger"] is None:
        st.info("No clean Shiomega setup on this tape/ticker right now — "
                "the cross may have fired without a following Chikou breakout, "
                "or the market is ranging. Quality over quantity: no momentum, no trade.")
        st.plotly_chart(ichimoku_fig(df, setup, side), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.plotly_chart(ichimoku_fig(df, setup, side), use_container_width=True,
                        config={"displayModeBar": False})
        long = side == "long"
        st.markdown(
            f'<span class="mono">①</span> {"Gold" if long else "Death"} cross '
            f'{"below" if long else "above"} the Kumo — heads-up only &nbsp;·&nbsp; '
            f'<span class="mono">②</span> we wait &nbsp;·&nbsp; '
            f'<span class="mono">③</span> Chikou breaks {"above" if long else "below"} '
            f'the past candles — the tide turns, enter next open &nbsp;·&nbsp; '
            f'<span class="mono">④</span> ride while the Kijun is respected',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    steps = [
        ("一", "Identify the prior trend",
         "For longs: price below Kijun and Kumo, Chikou under the candles. The setup is a reversal — no clear prior trend, no trade."),
        ("二", "TK cross in the right location",
         "Gold cross below the Kumo (buys) or death cross above it (sells). Crosses alone trap traders — this is only a heads-up."),
        ("三", "Wait for the Chikou breakout",
         "Entry only when the Chikou clears the past candles cleanly — not halfway, not inside the candles. Resistance cleared."),
        ("四", "Enter, protect, ride",
         "Fill next open. Stop below the Kijun or swing low (mirrored for shorts). Ride while the Kijun is respected."),
    ]
    cols = st.columns(4)
    for col, (num, head, body) in zip(cols, steps):
        col.markdown(f'<div class="step-card"><div class="n">{num}</div>'
                     f'<h4>{head}</h4><p>{body}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# helpers for the data tabs
# --------------------------------------------------------------------------- #
def style_signals(df: pd.DataFrame):
    disp = df.rename(columns={
        "ticker": "Ticker", "side": "Side", "date": "Date", "close": "Close",
        "entry": "Entry est.", "stop": "Stop", "risk_pct": "Risk %",
        "bars_since_cross": "Days since cross"})
    return disp


def run_live_scan(tickers: list[str]) -> pd.DataFrame:
    rows = []
    prog = st.progress(0.0, "scanning …")
    for k, t in enumerate(tickers, 1):
        try:
            raw = yf.download(t, period="2y", interval="1d",
                              auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            res = se.scan_last_bar(t, raw[["Open", "High", "Low", "Close", "Volume"]].dropna())
            if res:
                rows.append(res)
        except Exception:                         # noqa: BLE001
            pass
        prog.progress(k / len(tickers), f"scanning … {t}")
    prog.empty()
    cols = ["ticker", "status", "side", "date", "close", "entry", "stop",
            "risk_pct", "bars_since_cross"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# SIGNALS BOARD
# --------------------------------------------------------------------------- #
with tab_sig:
    src = st.radio("Signals source", ["Upload CSV", "Demo data"] +
                   (["Live scan"] if YF else []),
                   horizontal=True, label_visibility="collapsed")
    sig_df = None
    if src == "Upload CSV":
        up = st.file_uploader("Drop **shiomega_daily_signals.csv**", type=["csv"],
                              key="sig_up")
        st.markdown('<span class="note">Output of the daily screener · parsed in-app, '
                    'nothing leaves this session.</span>', unsafe_allow_html=True)
        if up is not None:
            sig_df = sc.normalise_signals(pd.read_csv(up))
    elif src == "Demo data":
        sig_df = sc.normalise_signals(sc.demo_signals())
    else:
        tk = st.text_input("Tickers (space/comma separated)",
                           "AAPL MSFT NVDA AMD SHOP.TO CNQ.TO SU.TO MRNA PYPL")
        if st.button("Run live scan", type="primary"):
            tickers = [x.strip().upper() for x in tk.replace(",", " ").split() if x.strip()]
            st.session_state.sig_live = run_live_scan(tickers)
        sig_df = st.session_state.get("sig_live")

    if sig_df is not None and len(sig_df):
        trig = sig_df[sig_df["status"] == "TRIGGERED"]
        watch = sig_df[sig_df["status"] == "WATCHLIST"]
        st.markdown(f"##### 🟡 TRIGGERED · {len(trig)}")
        st.caption("Clean Chikou breakout on the last daily close — actionable next open.")
        if len(trig):
            st.dataframe(style_signals(trig.sort_values("risk_pct")),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Nothing triggered on the last close. This setup doesn't happen every day.")
        st.markdown(f"##### 🔵 WATCHLIST · {len(watch)}")
        st.caption("Valid cross in place, waiting for the Chikou break — do not enter yet.")
        if len(watch):
            st.dataframe(style_signals(watch.sort_values("bars_since_cross")),
                         use_container_width=True, hide_index=True)
    elif sig_df is not None:
        st.info("No signals in this source.")


# --------------------------------------------------------------------------- #
# BACKTEST ANALYZER
# --------------------------------------------------------------------------- #
def render_backtest(metrics: dict):
    m = metrics
    st.markdown("##### Performance")
    r1 = st.columns(4)
    r1[0].metric("Net P&L", f"${m['net']:,.0f}")
    r1[1].metric("Trades", m["n"])
    r1[2].metric("Win rate", f"{m['win_rate']*100:.1f}%")
    r1[3].metric("Profit factor",
                 "∞" if m["profit_factor"] == np.inf else f"{m['profit_factor']:.2f}")
    r2 = st.columns(4)
    r2[0].metric("Expectancy / trade", f"${m['expectancy']:,.2f}")
    r2[1].metric("Expectancy (R)", "—" if m["expectancy_r"] is None else f"{m['expectancy_r']:.2f}R")
    r2[2].metric("Max drawdown", f"{m['max_dd']*100:.1f}%")
    r2[3].metric("CAGR", "—" if m["cagr"] is None else f"{m['cagr']*100:.1f}%")
    r3 = st.columns(4)
    r3[0].metric("Avg win", f"${m['avg_win']:,.0f}")
    r3[1].metric("Avg loss", f"${m['avg_loss']:,.0f}")
    r3[2].metric("Avg days held", "—" if m["avg_bars"] is None else f"{m['avg_bars']:.1f}")
    r3[3].metric("Long / Short", f"{m['long_n']} / {m['short_n']}")

    chips = " &nbsp; ".join(
        [f'<span style="color:{POS if m["long_pnl"]>=0 else NEG}">Longs ${m["long_pnl"]:,.0f}</span>',
         f'<span style="color:{POS if m["short_pnl"]>=0 else NEG}">Shorts ${m["short_pnl"]:,.0f}</span>']
        + [f'<span style="color:{MUT}">{k} × {v}</span>' for k, v in m["reasons"].items()])
    st.markdown(f'<div class="note" style="font-size:13px">{chips}</div>',
                unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Equity curve** · off $100,000")
        eq = m["curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=eq, x=list(range(len(eq))), fill="tozeroy",
                                 line=dict(color=GOLD, width=1.8),
                                 fillcolor="rgba(226,178,63,.08)"))
        fig.add_hline(y=eq.iloc[0], line=dict(color=FAINT, width=1, dash="dot"))
        lay = {**PLOT_LAYOUT, "height": 240}
        lay["yaxis"] = dict(gridcolor=GRID, zeroline=False,
                            range=[eq.min() * 0.999, eq.max() * 1.001])
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with cB:
        st.markdown("**R-multiple distribution** · 0.5R bins")
        rv = m["r_vals"]
        if len(rv):
            lo = np.floor(rv.min() / 0.5) * 0.5
            hi = np.ceil(rv.max() / 0.5) * 0.5
            edges = np.arange(lo, hi + 0.5, 0.5)
            counts, _ = np.histogram(rv, bins=edges)
            colors = [POS if e >= 0 else NEG for e in edges[:-1]]
            fig = go.Figure(go.Bar(x=edges[:-1] + 0.25, y=counts, marker_color=colors,
                                   width=0.46))
            fig.add_vline(x=0, line=dict(color=TEXT, width=1, dash="dash"))
            fig.update_layout(**{**PLOT_LAYOUT, "height": 240})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No R-multiples in this file.")

    st.markdown("**Trade log**")
    show = m["trades"].rename(columns={
        "ticker": "Ticker", "side": "Side", "entry_date": "Entry", "exit_date": "Exit",
        "entry": "In", "exit": "Out", "shares": "Shares", "pnl": "P&L",
        "r": "R", "reason": "Exit reason", "bars": "Days"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=440)


with tab_bt:
    bsrc = st.radio("Backtest source", ["Upload CSV", "Demo data"] +
                    (["Live backtest"] if YF else []),
                    horizontal=True, label_visibility="collapsed")
    metrics = None
    if bsrc == "Upload CSV":
        up = st.file_uploader("Drop **shiomega_daily_trades.csv**", type=["csv"], key="bt_up")
        st.markdown('<span class="note">Trade log from the daily backtest · metrics computed '
                    'off the $100K base. Column names are matched tolerantly.</span>',
                    unsafe_allow_html=True)
        if up is not None:
            t = sc.normalise_trades(pd.read_csv(up))
            if len(t):
                metrics = sc.compute_metrics(t)
            else:
                st.error("No usable trade rows — need a P&L column, or entry/exit prices + shares.")
    elif bsrc == "Demo data":
        metrics = sc.compute_metrics(sc.normalise_trades(sc.demo_trades()))
    else:
        tk = st.text_input("Tickers (space/comma separated)",
                           "AAPL MSFT NVDA AMD SHOP.TO CNQ.TO SU.TO MRNA", key="bt_tk")
        period = st.select_slider("History", ["1y", "2y", "5y", "10y"], value="5y")
        if st.button("Run live backtest", type="primary"):
            tickers = [x.strip().upper() for x in tk.replace(",", " ").split() if x.strip()]
            data = {}
            prog = st.progress(0.0, "downloading …")
            for k, t in enumerate(tickers, 1):
                try:
                    raw = yf.download(t, period=period, interval="1d",
                                      auto_adjust=False, progress=False)
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    data[t] = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                except Exception:                 # noqa: BLE001
                    pass
                prog.progress(k / len(tickers), f"downloading … {t}")
            prog.empty()
            trades, equity, summary = se.backtest_portfolio(data)
            st.session_state.bt_live = trades
        if st.session_state.get("bt_live") is not None and len(st.session_state.bt_live):
            metrics = sc.compute_metrics(sc.normalise_trades(st.session_state.bt_live))

    if metrics is not None and metrics["n"]:
        render_backtest(metrics)
    elif metrics is not None:
        st.info("No trades to analyze.")


# --------------------------------------------------------------------------- #
# RULES
# --------------------------------------------------------------------------- #
with tab_rules:
    cL, cR = st.columns(2)
    with cL:
        st.markdown("##### Long gates — 買い")
        st.markdown("""<table class="gate-tbl">
        <tr><td>L0</td><td>Prior downtrend: price below Kijun and Kumo, Chikou below the candles.</td></tr>
        <tr><td>L1</td><td>Tenkan gold-crosses Kijun while price closes <b>below the Kumo</b> — within the last 40 daily bars.</td></tr>
        <tr><td>L2</td><td>Tenkan sloping up over the last 3 bars.</td></tr>
        <tr><td>L3</td><td><b>Trigger:</b> close breaks cleanly above the 11-bar high window around the Chikou position — fresh, not halfway, not inside the candles.</td></tr>
        <tr><td>L4</td><td>Regime: close above SMA(50).</td></tr></table>""",
                    unsafe_allow_html=True)
    with cR:
        st.markdown("##### Short gates — 売り")
        st.markdown("""<table class="gate-tbl">
        <tr><td>S0</td><td>Prior uptrend: price above Kijun and Kumo, Chikou above the candles.</td></tr>
        <tr><td>S1</td><td>Tenkan death-crosses Kijun while price closes <b>above the Kumo</b> — within the last 40 daily bars.</td></tr>
        <tr><td>S2</td><td>Tenkan sloping down over the last 3 bars.</td></tr>
        <tr><td>S3</td><td><b>Trigger:</b> close breaks cleanly below the 11-bar low window around the Chikou position.</td></tr>
        <tr><td>S4</td><td>Regime: close below SMA(50).</td></tr></table>""",
                    unsafe_allow_html=True)

    st.markdown("---")
    cL2, cR2 = st.columns(2)
    with cL2:
        st.markdown("##### Entries & exits")
        st.markdown(f"""
- **Fill** — signal on daily close → enter next bar's open. No look-ahead.
- **Hard stop** — longs: min(Kijun, 10-bar swing low) − 0.1×ATR14; shorts mirrored. Intrabar, gap-aware.
- **Trail exit** — close through the Kijun against the position → exit next open.
- **Tie-break** — conservative: the stop resolves before any target on the same bar.
""")
    with cR2:
        st.markdown("##### Portfolio & universe")
        st.markdown(f"""
- **Capital** — $100,000 · $5,000/position · max 20 concurrent · one per underlier.
- **Commission** — $0.005/share, $1.00 minimum, both sides.
- **Universe** — NYSE / NASDAQ / TSX, market cap > $5B (yfinance EquityQuery).
- **Timeframe** — daily candles, Ichimoku 9 / 26 / 52, displacement 26.
""")

    st.markdown("---")
    cL3, cR3 = st.columns(2)
    with cL3:
        st.markdown("##### Stand aside when…")
        st.markdown("""
- Chikou is stuck inside the past candles — no clean break, no trade.
- The market is ranging, or there was no clear prior trend before the cross.
- The cross fires deep inside the Kumo — wait for a clearer transition near/outside the cloud.
- *No momentum, no trade.* Quality over quantity.
""")
    with cR3:
        st.markdown("##### Files & workflow")
        st.markdown("""
- `shiomega_daily_screener.py` → **shiomega_daily_signals.csv** (Signals board).
- `shiomega_daily_backtest.py` → **shiomega_daily_trades.csv** (Backtest analyzer).
- Cron the screener after the close, commit the CSV, or use the in-app **Live scan / backtest**.
- Weekly TK cross + daily Chikou breakout = the strongest expression of this setup.
""")

st.markdown(f'<div class="note" style="margin-top:24px">潮目が変わる — the tide has turned '
            '· Demo data is illustrative shape only, not results · Backtests are '
            'hypothetical; not investment advice.</div>', unsafe_allow_html=True)
