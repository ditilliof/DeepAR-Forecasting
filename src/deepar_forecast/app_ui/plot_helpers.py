"""
Shared plotting utilities for DeepAR forecast & backtest UIs.

All Plotly figure creation lives here so both apps use identical
y-axis scaling, colour palette, and layout logic.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Colour palette ──────────────────────────────────────────────────────
CLR_HIST = "#5b8fb9"       # muted steel-blue for historical
CLR_MEDIAN = "#00d4aa"     # teal-green accent for forecast median
CLR_ACTUAL = "#ff9f43"     # warm orange for actual (backtest)
CLR_80_FILL = "rgba(0,212,170,0.18)"
CLR_95_FILL = "rgba(0,212,170,0.08)"
CLR_CUTOFF = "#888888"
BG_COLOR = "#0e1117"       # Streamlit dark bg
PLOT_BG = "#161b22"        # slightly lighter card bg
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"


def _price_from_log_returns(last_close: float, lr: np.ndarray) -> np.ndarray:
    """P_t = P_0 · exp(cumsum(r))."""
    lr = np.asarray(lr, dtype=np.float64)
    return last_close * np.exp(np.cumsum(lr))


def _robust_yrange(
    *series_list,
    mode: str = "robust",
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    pad_frac: float = 0.05,
):
    """
    Compute a y-axis range that avoids extreme outliers.

    Parameters
    ----------
    *series_list : array-like
        One or more arrays of y-values (NaN / inf are dropped).
    mode : str
        ``"robust"`` – use percentiles (default).
        ``"full"``   – use raw min/max.
    lo_pct, hi_pct : float
        Percentile bounds for ``"robust"`` mode.
    pad_frac : float
        Fractional padding on each side.

    Returns
    -------
    (y_lo, y_hi) : tuple[float, float]
    """
    combined = np.concatenate(
        [np.asarray(s, dtype=np.float64).ravel() for s in series_list if s is not None and len(s)]
    )
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return (0, 1)

    if mode == "full":
        lo, hi = float(combined.min()), float(combined.max())
    else:
        lo = float(np.percentile(combined, lo_pct))
        hi = float(np.percentile(combined, hi_pct))

    pad = max((hi - lo) * pad_frac, 1e-4)
    return lo - pad, hi + pad


# ──────────────────────────────────────────────────────────────────────────
# Common dark-dashboard Plotly layout
# ──────────────────────────────────────────────────────────────────────────
def _base_layout(title: str, height: int = 540) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=18, color=TEXT_COLOR)),
        xaxis=dict(
            title="Date / Time",
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=TEXT_COLOR,
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=TEXT_COLOR,
        ),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        hovermode="x unified",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_COLOR),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
    )


# ──────────────────────────────────────────────────────────────────────────
# Main forecast plot (used by app_ui/main.py)
# ──────────────────────────────────────────────────────────────────────────
def create_forecast_plot(
    historical_data: List[Dict],
    forecast_data: Dict,
    symbol: str,
    timeframe: str,
    scale_mode: str = "robust",
) -> go.Figure:
    """Build a Plotly figure with historical close + forecast price overlay.

    Parameters
    ----------
    historical_data : list[dict]
        Raw OHLCV rows (from ``/data/ohlcv``).
    forecast_data : dict
        Keys: forecast_dates, median, lower_80, upper_80, lower_95, upper_95.
    symbol, timeframe : str
        Labels.
    scale_mode : str
        ``"robust"`` for percentile-bounded y-axis (default);
        ``"full"`` for raw min/max.
    """
    hist_df = pd.DataFrame(historical_data)
    if hist_df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(f"{symbol} — No historical data"))
        return fig

    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.sort_values("timestamp")
    hist_close = hist_df["close"].values.astype(np.float64)

    # ── Parse forecast ─────────────────────────────────────────────────
    forecast_dates = pd.to_datetime(forecast_data.get("forecast_dates", []))
    median_lr = np.array(forecast_data.get("median", []), dtype=np.float64)

    if median_lr.size == 0 or len(forecast_dates) == 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_close,
            mode="lines", name="Historical",
            line=dict(color=CLR_HIST, width=2),
        ))
        fig.update_layout(**_base_layout(f"{symbol} — Historical only ({timeframe})"))
        return fig

    def _safe(key, fb=None):
        arr = forecast_data.get(key, [])
        if arr is None or (isinstance(arr, list) and len(arr) == 0):
            arr = forecast_data.get(fb, []) if fb else []
        if arr is None or (isinstance(arr, list) and len(arr) == 0):
            return None
        return np.array(arr, dtype=np.float64)

    lower_80_lr = _safe("lower_80", "lower_95")
    upper_80_lr = _safe("upper_80", "upper_95")
    lower_95_lr = _safe("lower_95")
    upper_95_lr = _safe("upper_95")

    last_close = float(hist_close[-1])
    median_p = _price_from_log_returns(last_close, median_lr)

    has_80 = lower_80_lr is not None and upper_80_lr is not None
    has_95 = lower_95_lr is not None and upper_95_lr is not None

    l80_p = _price_from_log_returns(last_close, lower_80_lr) if has_80 else None
    u80_p = _price_from_log_returns(last_close, upper_80_lr) if has_80 else None
    l95_p = _price_from_log_returns(last_close, lower_95_lr) if has_95 else None
    u95_p = _price_from_log_returns(last_close, upper_95_lr) if has_95 else None

    # ── Y-axis range ───────────────────────────────────────────────────
    y_lo, y_hi = _robust_yrange(
        hist_close, median_p,
        l80_p, u80_p, l95_p, u95_p,
        mode=scale_mode,
    )

    # ── Build figure ───────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_close,
        mode="lines", name="Historical",
        line=dict(color=CLR_HIST, width=2),
    ))

    # 95 % band
    if has_95:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=u95_p,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=l95_p,
            mode="lines", name="95 % Interval",
            line=dict(width=0), fill="tonexty", fillcolor=CLR_95_FILL,
        ))

    # 80 % band
    if has_80:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=u80_p,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=l80_p,
            mode="lines", name="80 % Interval",
            line=dict(width=0), fill="tonexty", fillcolor=CLR_80_FILL,
        ))

    # Median
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=median_p,
        mode="lines+markers", name="Forecast (Median)",
        line=dict(color=CLR_MEDIAN, width=3, dash="dash"),
        marker=dict(size=5),
    ))

    # Connection line
    fig.add_trace(go.Scatter(
        x=[hist_df["timestamp"].iloc[-1], forecast_dates[0]],
        y=[last_close, median_p[0]],
        mode="lines",
        line=dict(color=CLR_MEDIAN, width=1, dash="dot"),
        showlegend=False,
    ))

    layout = _base_layout(f"{symbol} Forecast ({timeframe})")
    layout["yaxis"]["range"] = [y_lo, y_hi]
    fig.update_layout(**layout)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# Backtest plot (used by app_ui/main_backtest.py)
# ──────────────────────────────────────────────────────────────────────────
def create_backtest_plot(
    hist_df: pd.DataFrame,
    forecast_data: Dict,
    actual_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cutoff_date: str,
    scale_mode: str = "robust",
) -> go.Figure:
    """Build Plotly chart: history + forecast intervals + actual realised path."""
    fig = go.Figure()
    last_close = float(hist_df["close"].iloc[-1])

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df["close"],
        mode="lines", name="Historical",
        line=dict(color=CLR_HIST, width=2),
    ))

    fc_dates = forecast_data["dates"]
    median = forecast_data["median"]
    lower_80 = forecast_data["lower_80"]
    upper_80 = forecast_data["upper_80"]
    lower_95 = forecast_data["lower_95"]
    upper_95 = forecast_data["upper_95"]

    # Convert to price if log-returns
    if abs(float(np.asarray(median).mean())) < 0.5:
        median_p = _price_from_log_returns(last_close, median)
        l80_p = _price_from_log_returns(last_close, lower_80)
        u80_p = _price_from_log_returns(last_close, upper_80)
        l95_p = _price_from_log_returns(last_close, lower_95)
        u95_p = _price_from_log_returns(last_close, upper_95)
    else:
        median_p = np.asarray(median, dtype=np.float64)
        l80_p = np.asarray(lower_80, dtype=np.float64)
        u80_p = np.asarray(upper_80, dtype=np.float64)
        l95_p = np.asarray(lower_95, dtype=np.float64)
        u95_p = np.asarray(upper_95, dtype=np.float64)

    # 95 % band
    fig.add_trace(go.Scatter(x=fc_dates, y=u95_p, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=l95_p, mode="lines", name="95 % Interval",
        line=dict(width=0), fill="tonexty", fillcolor=CLR_95_FILL,
    ))
    # 80 % band
    fig.add_trace(go.Scatter(x=fc_dates, y=u80_p, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=l80_p, mode="lines", name="80 % Interval",
        line=dict(width=0), fill="tonexty", fillcolor=CLR_80_FILL,
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=fc_dates, y=median_p, mode="lines+markers", name="Forecast (Median)",
        line=dict(color=CLR_MEDIAN, width=3, dash="dash"), marker=dict(size=5),
    ))

    # Actual
    actual_close = None
    if actual_df is not None and not actual_df.empty:
        actual_close = actual_df["close"].values.astype(np.float64)
        fig.add_trace(go.Scatter(
            x=actual_df["timestamp"], y=actual_close,
            mode="lines+markers", name="Actual",
            line=dict(color=CLR_ACTUAL, width=3), marker=dict(size=5),
        ))

    # Cutoff vertical line
    cutoff_ts = pd.Timestamp(cutoff_date)
    fig.add_vline(x=cutoff_ts, line_dash="dot", line_color=CLR_CUTOFF, annotation_text="Cutoff",
                  annotation_font_color=TEXT_COLOR)

    # y-axis
    y_lo, y_hi = _robust_yrange(
        hist_df["close"].values, median_p, l80_p, u80_p, l95_p, u95_p, actual_close,
        mode=scale_mode,
    )

    layout = _base_layout(f"{symbol} Backtest — Forecast vs Actual ({timeframe})", height=560)
    layout["yaxis"]["range"] = [y_lo, y_hi]
    fig.update_layout(**layout)
    return fig
