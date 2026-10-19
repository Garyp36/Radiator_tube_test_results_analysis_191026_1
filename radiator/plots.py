"""Plotly figures for the single-experiment and comparison views."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .fitting import predict_exponential, predict_linear

COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def _layout(fig: go.Figure, title: str, ytitle: str) -> go.Figure:
    fig.update_layout(title=title, xaxis_title="Elapsed time (min)", yaxis_title=ytitle,
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02),
                      margin=dict(l=40, r=20, t=70, b=40), template="plotly_white")
    return fig


def temperature_plot(p: pd.DataFrame, linear_fit: Optional[Dict] = None, exp_fit: Optional[Dict] = None,
                     show_fit: bool = True) -> go.Figure:
    fig = go.Figure()
    series = [("T0_C", "T0 ambient (°C)"), ("T1_C", "T1 heated vessel (°C)"), ("T2_C", "T2 inlet end (°C)"),
              ("T3_C", "T3 outlet end (°C)"), ("Tmean_C", "Tmean = (T2+T3)/2 (°C)")]
    for col, label in series:
        if col in p.columns and p[col].notna().any():
            fig.add_trace(go.Scatter(x=p["elapsed_min"], y=p[col], mode="lines+markers", name=label,
                                     hovertemplate="%{y:.2f} °C"))
    if show_fit:
        t_dense = np.linspace(p["elapsed_min"].min(), p["elapsed_min"].max(), 200)
        if linear_fit and linear_fit.get("ok"):
            fig.add_trace(go.Scatter(x=t_dense, y=predict_linear(linear_fit, t_dense), mode="lines",
                                     name="Linear fit of Tmean", line=dict(dash="dash", color="black")))
        if exp_fit and exp_fit.get("ok"):
            # Exponential fit is for θ; convert back to temperature using per-row ambient (interpolated).
            ambient = np.interp(t_dense, p["elapsed_min"], p["ambient_used_C"])
            fig.add_trace(go.Scatter(x=t_dense, y=predict_exponential(exp_fit, t_dense) + ambient, mode="lines",
                                     name="Exponential fit (θ + ambient)", line=dict(dash="dot", color="gray")))
    return _layout(fig, "Temperature versus elapsed time", "Temperature (°C)")


def theta_plot(p: pd.DataFrame, exp_fit: Optional[Dict] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["elapsed_min"], y=p["theta_C"], mode="lines+markers",
                             name="θ = Tmean − T0 (°C)", hovertemplate="%{y:.2f} °C"))
    if exp_fit and exp_fit.get("ok"):
        t_dense = np.linspace(p["elapsed_min"].min(), p["elapsed_min"].max(), 200)
        fig.add_trace(go.Scatter(x=t_dense, y=predict_exponential(exp_fit, t_dense), mode="lines",
                                 name=f"Exponential fit (R² = {exp_fit['r2']:.3f})",
                                 line=dict(dash="dash", color="black")))
    return _layout(fig, "Ambient-corrected temperature excess", "Temperature excess θ (°C)")


def energy_plot(p: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["elapsed_min"], y=p["cumulative_energy_J"], mode="lines+markers",
                             name="Cumulative sensible energy removed (J)", hovertemplate="%{y:.1f} J"))
    return _layout(fig, "Cumulative sensible energy removed from water", "Energy (J)")


def rates_plot(p: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label in (("rate_T2_C_per_h", "T2 interval cooling rate"), ("rate_T3_C_per_h", "T3 interval cooling rate"),
                       ("rate_Tmean_C_per_h", "Tmean interval cooling rate")):
        fig.add_trace(go.Scatter(x=p["elapsed_min"], y=p[col], mode="lines+markers", name=f"{label} (°C/h)",
                                 hovertemplate="%{y:.2f} °C/h", connectgaps=False))
    fig = _layout(fig, "Interval cooling rates (interval-average estimates, plotted at interval end)",
                  "Cooling rate (°C/h)")
    return fig


def comparison_plot(processed_by_name: Dict[str, pd.DataFrame], columns, ytitle: str, title: str) -> go.Figure:
    """Overlay ``columns`` (list of (col, suffix)) for every experiment with distinct colour/dash."""
    fig = go.Figure()
    for i, (name, p) in enumerate(processed_by_name.items()):
        colour = COLOURS[i % len(COLOURS)]
        for j, (col, suffix) in enumerate(columns):
            if col in p.columns and p[col].notna().any():
                fig.add_trace(go.Scatter(x=p["elapsed_min"], y=p[col], mode="lines+markers",
                                         name=f"{name} – {suffix}", legendgroup=name,
                                         line=dict(color=colour, dash=DASHES[j % len(DASHES)]),
                                         marker=dict(size=5)))
    return _layout(fig, title, ytitle)
