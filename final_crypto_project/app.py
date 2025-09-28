import numpy as np, pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from arch import arch_model
import config
from src.data.sqlite_io import load_ohlcv_from_sqlite
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=False)


# ---- Keep heavy models server-side (not in dcc.Store) ----
GLOBAL_MODELS = {"var_res": None, "lag_order": None, "g_mod": None}


app = Dash(__name__)
server = app.server  # <- this is what gunicorn/app servers look for


# Load full DF once (safe fallback to empty)
try:
    DF_FULL = load_ohlcv_from_sqlite(config.TABLE, config.DB_PATH)
except Exception:
    DF_FULL = pd.DataFrame(columns=["open","high","low","close","volume"])

app.layout = html.Div([
    html.Div([
        html.H2("Streaming VAR–GARCH Backtest"),
        html.Div([
            html.Label("Start:"), dcc.Input(id="start", type="text", placeholder="YYYY-MM-DD", style={"width":"160px"}),
            html.Label("End:", style={"marginLeft":"12px"}), dcc.Input(id="end", type="text", placeholder="YYYY-MM-DD", style={"width":"160px"}),
            html.Button("Load Range", id="btn-load", n_clicks=0, style={"marginLeft":"12px"}),
            html.Button("Start", id="btn-start", n_clicks=0, style={"marginLeft":"12px"}),
            html.Button("Pause", id="btn-pause", n_clicks=0, style={"marginLeft":"6px"}),
            html.Button("Reset", id="btn-reset", n_clicks=0, style={"marginLeft":"6px"}),
        ]),
        html.Div(id="status", style={"marginTop":"8px","color":"#6b7280"})
    ], className="card"),
    dcc.Interval(id="tick", interval=config.INTERVAL_MS, disabled=True),
    dcc.Store(id="store-df"),
    dcc.Store(id="store-i", data=0),
    dcc.Store(id="store-logs", data={"ts_x":[],"close_y":[],"cummax_y":[],"logret_y":[],"realvol_y":[],"fcst_x":[],"fcst_y":[],"cstrategy_y":[],"creturns_y":[],"mean_fcst_x":[],"mean_fcst_y":[] }),
    dcc.Store(id="store-state", data={"hold_ttl":0,"cooldown":0,"position":1,"cum_asset":0.0,"cum_strat":0.0,"s_hist":[],"n":0}),
    dcc.Store(id="store-train", data={"timestamp":[],"open":[],"high":[],"low":[],"close":[],"volume":[]}),
    dcc.Graph(id="fig-candle"),
    dcc.Graph(id="fig-price"),
    dcc.Graph(id="fig-cum"),
    dcc.Graph(id="fig-logs"),
    dcc.Graph(id="fig-vol"),
], className="container")

# ---------- helpers ----------
def fit_var_garch(y_vars, lookback):
    var = VAR(y_vars.iloc[-lookback:])
    var_res = var.fit(maxlags=6, trend='n')
    lag_order = var_res.k_ar
    resids = var_res.resid['close'].dropna()
    resids = (resids * 1000).iloc[-lookback:]
    g_mod = arch_model(resids, p=1, q=1, vol='GARCH', mean='Zero', dist='t', rescale=False).fit(disp='off')
    return var_res, lag_order, g_mod

def garch_block_vol(g_mod):
    f = g_mod.forecast(horizon=6)
    var_path = f.variance.iloc[-1].values
    return float(np.sqrt(var_path.sum()) / 1000.0)

def ewma_sigma_cut(rv, lam=0.94, base=0.008, k=1.5, min_obs=30):
    rv = rv.dropna()
    if len(rv) < min_obs:
        return base
    alpha = 1 - lam
    m = rv.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    v = rv.ewm(alpha=alpha, adjust=False).var(bias=False).iloc[-1]
    return max(base, m + k*np.sqrt(v))

# ---------- callbacks ----------
@callback(Output("store-df","data"), Output("status","children"),
          Input("btn-load","n_clicks"), State("start","value"), State("end","value"), prevent_initial_call=True)
def load_range(n, start, end):
    df = load_ohlcv_from_sqlite(config.TABLE, config.DB_PATH, start=start, end=end)
    msg = f"Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}"
    return {"index": df.index.astype(str).tolist(), "data": df.to_dict(orient="list")}, msg

@callback(Output("tick","disabled"),
          Input("btn-start","n_clicks"), Input("btn-pause","n_clicks"),
          State("tick","disabled"), prevent_initial_call=True)
def start_pause(n_start, n_pause, disabled):
    if not ctx.triggered_id:
        return disabled
    if ctx.triggered_id == "btn-start":
        return False
    if ctx.triggered_id == "btn-pause":
        return True
    return disabled

@callback(Output("store-i","data"), Output("store-logs","data"), Output("store-state","data"), Output("store-train","data"),
          Input("tick","n_intervals"),
          State("store-df","data"), State("store-i","data"), State("store-logs","data"), State("store-state","data"), State("store-train","data"),
          prevent_initial_call=True)
def on_tick(_, store_df, i, logs, state, train):
    df = pd.DataFrame(store_df["data"], index=pd.to_datetime(store_df["index"])) if (store_df and store_df.get("data")) else DF_FULL.copy()
    if df.empty: 
        return i, logs, state, train

    i = 0 if i is None else i
    i = min(i+1, len(df)-1)
    partial = df.iloc[:i+1].copy()
    latest = partial.iloc[-1]; tstamp = partial.index[-1]
    partial["cummax"] = partial["close"].cummax()

    vol = partial["volume"].replace(0, np.nan)
    log_chg_close = np.log(partial["close"]/partial["close"].shift(1))
    r_last = float(log_chg_close.iloc[-1]) if pd.notna(log_chg_close.iloc[-1]) else 0.0

    # accrual
    state["cum_asset"] += r_last
    state["cum_strat"] += state["position"] * r_last
    logs["creturns_y"].append(float(np.exp(state["cum_asset"])))
    logs["cstrategy_y"].append(float(np.exp(state["cum_strat"])))

    # realized vol (6h)
    realised = float(np.sqrt((log_chg_close**2).rolling(6).sum()).iloc[-1]) if len(log_chg_close) >= 6 else None

    # book-keeping
    logs["ts_x"].append(tstamp.isoformat())
    logs["close_y"].append(float(latest.close))
    logs["logret_y"].append(r_last)
    logs["realvol_y"].append(realised)
    logs["cummax_y"].append(float(partial["cummax"].iloc[-1]))

    # train log changes
    for col, series in [("open", partial["open"]), ("high", partial["high"]), ("low", partial["low"]), ("close", partial["close"]), ("volume", vol)]:
        change = np.log(series/series.shift(1)).iloc[-1]
        x = float(change) if pd.notna(change) and np.isfinite(change) else None
        train[col].append(x)
    train["timestamp"].append(tstamp.isoformat())

    # model update/forecast
    df_train = pd.DataFrame(train).dropna()
    mu_signal = np.nan; vol_6h = np.nan
    min_sample = config.MIN_SAMPLE
    lookback = int(config.LOOKBACK_FRAC * len(df_train)) if len(df_train) else 0
    N_update = config.N_UPDATE

    if len(df_train) >= min_sample and state["n"] % N_update == 0:
        y_vars = df_train[["open","high","low","close","volume"]].dropna().iloc[-lookback:]
        y_vars = y_vars.replace([np.inf,-np.inf], np.nan).dropna()
        if len(y_vars) >= 10:
            var_res, lag_order, g_mod = fit_var_garch(y_vars, lookback=max(10,lookback))
            GLOBAL_MODELS["lag_order"] = int(lag_order)
            GLOBAL_MODELS["var_res"] = var_res
            GLOBAL_MODELS["g_mod"] = g_mod

    if GLOBAL_MODELS.get("lag_order"):
        y_vars = df_train[["open","high","low","close","volume"]].dropna().iloc[-lookback:]
        if len(y_vars) >= GLOBAL_MODELS["lag_order"]:
            # re-fit quickly on sliding window to forecast mean
            var_res2 = VAR(y_vars).fit(maxlags=6, trend='n')
            GLOBAL_MODELS["lag_order"] = var_res2.k_ar
            mu_path = var_res2.forecast(y_vars.values[-GLOBAL_MODELS["lag_order"]:], steps=6)
            idx_close = list(y_vars.columns).index("close")
            mu_vec = mu_path[:, idx_close]
            mu_signal = float(np.sum(mu_vec))
            g_mod = GLOBAL_MODELS.get("g_mod", None)
            if g_mod is not None:
                vol_6h = garch_block_vol(g_mod)

    # dynamic sigma cut
    rv_series = pd.Series([x for x in logs["realvol_y"] if x is not None], dtype=float)
    sigma_cut = ewma_sigma_cut(rv_series)

    # score & adaptive eps
    if np.isfinite(mu_signal) and np.isfinite(vol_6h) and vol_6h > 0:
        s = float(mu_signal / vol_6h)
        if np.isfinite(s):
            s_hist = pd.Series(state["s_hist"] + [s])
            state["s_hist"] = s_hist.tail(2000).tolist()
    else:
        s = float("nan")
    s_hist_abs = np.abs(pd.Series(state["s_hist"])).dropna() if len(state["s_hist"]) else pd.Series(dtype=float)
    s_eps = float(s_hist_abs.quantile(0.40)) if len(s_hist_abs) >= 200 else 0.10

    # trailing stop logic
    close_price = float(latest.close)
    cummax = float(partial["cummax"].iloc[-1])
    real_vol_block_now = float(np.sqrt((pd.Series(logs["logret_y"])**2).rolling(6).sum()).iloc[-1]) if len(logs["logret_y"]) >= 6 else 0.0
    dd_pct = float(np.clip(0.015 + 2.0*real_vol_block_now, 0.015, 0.08))
    trail_threshold = cummax*(1-dd_pct)
    trail_break = close_price < trail_threshold

    # cooldown & hold
    if state["cooldown"]>0: state["cooldown"] -= 1
    if state["hold_ttl"]>0: state["hold_ttl"] = max(state["hold_ttl"]-1,0)

    new_pos = state["position"]
    if np.isfinite(mu_signal) and np.isfinite(vol_6h):
        s_ok_long = (not np.isnan(s)) and (s > s_eps)
        emergency_flat = trail_break and ((not np.isnan(s) and s <= s_eps/2) or mu_signal < -abs(s_eps)*vol_6h)
        if emergency_flat and state["cooldown"]==0:
            new_pos = 0
        elif state["hold_ttl"]>0:
            new_pos = 1
            if s_ok_long: state["hold_ttl"] = 6
        else:
            if s_ok_long and state["cooldown"]==0:
                new_pos = 1; state["hold_ttl"]=6
            elif (not np.isnan(s)) and (s < -s_eps) and (vol_6h > sigma_cut) and state["cooldown"]==0:
                new_pos = 0

    if int(new_pos) != int(state["position"]):
        state["cooldown"] = 2
    state["position"] = int(new_pos)

    # store forecasts for plotting (6h ahead)
    if np.isfinite(vol_6h) and np.isfinite(mu_signal):
        future_ts = tstamp + pd.Timedelta(hours=6)
        logs["fcst_x"].append(future_ts.isoformat()); logs["fcst_y"].append(float(vol_6h))
        logs["mean_fcst_x"].append(future_ts.isoformat()); logs["mean_fcst_y"].append(float(mu_signal))

    state["n"] += 1
    return i, logs, state, train

@callback(Output("fig-price","figure"), Output("fig-cum","figure"), Output("fig-logs","figure"), Output("fig-vol","figure"),
          Input("store-logs","data"))
def render_figs(logs):
    ts = pd.to_datetime(logs["ts_x"]) if logs and logs.get("ts_x") else pd.to_datetime([])
    fig1 = go.Figure()
    if len(ts):
        fig1.add_scatter(x=ts, y=logs["close_y"], mode="lines", name="Close")
        fig1.add_scatter(x=ts, y=logs["cummax_y"], mode="lines", name="CumMax")
    fig1.update_layout(title="Close & Trailing Max", height=320, margin=dict(l=10,r=10,t=40,b=10))

    fig2 = go.Figure()
    if len(ts):
        fig2.add_scatter(x=ts, y=logs["creturns_y"], mode="lines", name="Buy & Hold")
        fig2.add_scatter(x=ts, y=logs["cstrategy_y"], mode="lines", name="VAR-GARCH Mean Str", opacity=0.7)
    fig2.update_layout(title="Cumulative Returns", height=300, margin=dict(l=10,r=10,t=40,b=10))

    fig3 = go.Figure()
    if len(ts):
        fig3.add_scatter(x=ts, y=logs["logret_y"], mode="lines", name="log_ret")
    if logs.get("mean_fcst_x"):
        fx = pd.to_datetime(logs["mean_fcst_x"]); fy = logs["mean_fcst_y"]
        fig3.add_scatter(x=fx, y=fy, mode="lines", name="mu 6h-ahead", opacity=0.7)
    fig3.add_hline(y=0, line_width=1, opacity=0.3)
    fig3.update_layout(title="Hourly Log-Returns & 6h μ Forecast", height=300, margin=dict(l=10,r=10,t=40,b=10))

    fig4 = go.Figure()
    if len(ts):
        fig4.add_scatter(x=ts, y=logs["realvol_y"], mode="lines", name="Realised σ (6h)")
    if logs.get("fcst_x"):
        vx = pd.to_datetime(logs["fcst_x"]); vy = logs["fcst_y"]
        fig4.add_scatter(x=vx, y=vy, mode="lines", name="σ 6h-ahead", opacity=0.8)
    fig4.update_layout(title="Volatility: realised vs 6h forecast", height=320, margin=dict(l=10,r=10,t=40,b=10))

    return fig1, fig2, fig3, fig4

@callback(
    Output("fig-candle", "figure"),
    Input("store-df", "data"),
    Input("store-i", "data"),
    prevent_initial_call=False
)
def render_candle(store_df, i):
    # Rebuild DF from store; fallback to full DF if store is empty
    df = (pd.DataFrame(store_df["data"], index=pd.to_datetime(store_df["index"]))
          if (store_df and store_df.get("data")) else DF_FULL.copy())
    if df.empty:
        return go.Figure()

    i = 0 if i in (None, 0) else min(int(i), len(df)-1)
    partial = df.iloc[:i+1].copy()
    if partial.empty:
        return go.Figure()

    # Cufflinks QuantFig wants OHLC with capitalized names and a DatetimeIndex
    ohlc = partial[["open","high","low","close"]].rename(
        columns={"open":"Open", "high":"High", "low":"Low", "close":"Close"}
    )
    try:
        # Preferred: Cufflinks QuantFig (nice default layout)
        qf = cf.QuantFig(
            ohlc,
            title="BTC 1h Candlestick",
            legend="top",
            name="BTC",
            up_color="green",
            down_color="red",
        )
        fig = qf.iplot(asFigure=True)
    except Exception:
        # Fallback: native Plotly candlestick (if cufflinks not available)
        fig = go.Figure(go.Candlestick(
            x=ohlc.index,
            open=ohlc["Open"],
            high=ohlc["High"],
            low=ohlc["Low"],
            close=ohlc["Close"],
            name="BTC"
        ))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=None, yaxis_title=None
    )
    return fig


# Some hosts want this name instead of `server`
application = server

if __name__ == "__main__":
    app.run_server(debug=True)