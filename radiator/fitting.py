"""Curve fitting: linear temperature trend and exponential (lumped-capacitance style) decay.

Both models are purely descriptive. A good fit does not by itself prove the
physical mechanism; the exponential form is what Newtonian cooling with a
constant ambient would produce, but stored energy in tube walls, sensor lag and
heat exchange with the vessel are not represented.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
except ImportError:  # pragma: no cover
    curve_fit = None

MIN_POINTS_LINEAR = 3
MIN_POINTS_EXP = 4


def _goodness(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return {"r2": r2, "rmse": rmse}


def _empty(model: str, message: str) -> Dict[str, object]:
    return {"model": model, "ok": False, "message": message, "r2": float("nan"), "rmse": float("nan"),
            "n_points": 0, "equation": "—", "fitted": None}


def exponential_model(t_min: np.ndarray, A: float, k: float) -> np.ndarray:
    """θ(t) = A · exp(−k · t), t in minutes, k in 1/min."""
    return A * np.exp(-k * t_min)


def fit_linear(t_min, temperature_C) -> Dict[str, object]:
    """Fit T = intercept + slope·t (t in minutes) by ordinary least squares."""
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(temperature_C, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < MIN_POINTS_LINEAR:
        return _empty("linear", f"Need at least {MIN_POINTS_LINEAR} valid points for a linear fit.")
    if np.ptp(t) <= 0:
        return _empty("linear", "All observations share the same time; cannot fit.")
    slope, intercept = np.polyfit(t, y, 1)
    y_hat = intercept + slope * t
    gof = _goodness(y, y_hat)
    fitted = pd.DataFrame({"elapsed_min": t, "observed_C": y, "predicted_C": y_hat, "residual_C": y - y_hat})
    return {
        "model": "linear", "ok": True, "message": "",
        "intercept_C": float(intercept), "slope_C_per_min": float(slope),
        "cooling_rate_C_per_h": float(-slope * 60.0),
        "equation": f"Tmean = {intercept:.4g} + ({slope:.4g}) × t  [t in min]",
        "n_points": int(len(t)), "fitted": fitted, **gof,
    }


def fit_exponential(t_min, theta_C) -> Dict[str, object]:
    """Fit θ = A·exp(−k·t) to ambient-corrected excess temperature with SciPy curve_fit.

    Constraints: A > 0 (bounded away from zero), k within [-1, 10] 1/min so that
    unphysical but data-supported slightly negative decay constants are reported
    rather than hidden. Only points with θ > 0 contribute to the log-space
    initial guess; all finite points enter the nonlinear fit.
    """
    if curve_fit is None:
        return _empty("exponential", "SciPy is not installed; exponential fitting unavailable.")
    t = np.asarray(t_min, dtype=float)
    y = np.asarray(theta_C, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < MIN_POINTS_EXP:
        return _empty("exponential", f"Need at least {MIN_POINTS_EXP} valid points for an exponential fit.")
    if np.ptp(t) <= 0:
        return _empty("exponential", "All observations share the same time; cannot fit.")
    positive = y > 0
    if positive.sum() < 2:
        return _empty("exponential", "Temperature excess must be positive for at least two readings to fit a decay.")
    # Initial guess from a log-linear regression on the positive excess values.
    try:
        slope, intercept = np.polyfit(t[positive], np.log(y[positive]), 1)
        p0 = [float(np.exp(intercept)), float(max(-slope, 1e-6))]
    except Exception:
        p0 = [float(max(y[0], 1e-3)), 1e-3]
    t_shift = t - t[0]
    try:
        popt, _ = curve_fit(exponential_model, t_shift, y, p0=p0, bounds=([1e-9, -1.0], [np.inf, 10.0]), maxfev=20000)
    except Exception as exc:
        return _empty("exponential", f"Nonlinear fit did not converge: {exc}")
    A, k = float(popt[0]), float(popt[1])
    y_hat = exponential_model(t_shift, A, k)
    if not np.all(np.isfinite(y_hat)):
        return _empty("exponential", "Fit produced non-finite predictions.")
    gof = _goodness(y, y_hat)
    tau = 1.0 / k if k > 0 else float("nan")
    fitted = pd.DataFrame({"elapsed_min": t, "observed_theta_C": y, "predicted_theta_C": y_hat,
                           "residual_C": y - y_hat})
    return {
        "model": "exponential", "ok": True, "message": "",
        "A_C": A, "k_per_min": k, "tau_min": tau, "t0_min": float(t[0]),
        "equation": f"θ = {A:.4g} × exp(−{k:.4g} × (t − {t[0]:.4g}))  [t in min]",
        "n_points": int(len(t)), "fitted": fitted, **gof,
    }


def predict_exponential(fit: Dict[str, object], t_min) -> Optional[np.ndarray]:
    if not fit.get("ok"):
        return None
    t = np.asarray(t_min, dtype=float) - float(fit["t0_min"])
    return exponential_model(t, fit["A_C"], fit["k_per_min"])


def predict_linear(fit: Dict[str, object], t_min) -> Optional[np.ndarray]:
    if not fit.get("ok"):
        return None
    t = np.asarray(t_min, dtype=float)
    return fit["intercept_C"] + fit["slope_C_per_min"] * t
