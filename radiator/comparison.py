"""Experiment comparison: summary table and common-criterion comparisons."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .calculations import first_downward_crossing, value_at_time

COMPARISON_COLUMNS = [
    ("design", "Experiment / design"),
    ("air_velocity_m_s", "Air velocity (m/s)"),
    ("fan_pwm_pct", "Fan PWM (%)"),
    ("fan_rpm", "Fan RPM"),
    ("duration_min", "Analysis duration (min)"),
    ("water_mass_kg", "Water mass (kg)"),
    ("Tmean_initial_C", "Initial Tmean (°C)"),
    ("Tmean_final_C", "Final Tmean (°C)"),
    ("Tmean_drop_C", "Mean temperature drop (°C)"),
    ("rate_T2_C_per_h", "T2 cooling rate (°C/h)"),
    ("rate_T3_C_per_h", "T3 cooling rate (°C/h)"),
    ("rate_Tmean_C_per_h", "Tmean cooling rate (°C/h)"),
    ("energy_removed_J", "Sensible energy removed (J)"),
    ("avg_heat_loss_W", "Average net heat-loss rate (W)"),
    ("exposed_area_m2", "Exposed area (m²)"),
    ("apparent_heat_flux_W_m2", "Apparent average heat flux (W/m²)"),
    ("gap_mean_C", "Mean |T2−T3| (°C)"),
    ("gap_max_C", "Max |T2−T3| (°C)"),
    ("linear_r2", "Linear fit R²"),
    ("linear_rmse_C", "Linear fit RMSE (°C)"),
    ("exp_r2", "Exponential fit R²"),
    ("exp_rmse_C", "Exponential fit RMSE (°C)"),
    ("ambient_source", "Ambient source"),
]


def build_comparison_table(summaries: List[Dict[str, object]], fits: Dict[str, Dict[str, Dict[str, object]]]) -> pd.DataFrame:
    """Combine per-experiment summaries and fit metrics into one table (one row per experiment)."""
    rows = []
    for s in summaries:
        row = dict(s)
        f = fits.get(s["experiment"], {})
        lin, expo = f.get("linear", {}), f.get("exponential", {})
        row["linear_r2"] = lin.get("r2", float("nan"))
        row["linear_rmse_C"] = lin.get("rmse", float("nan"))
        row["exp_r2"] = expo.get("r2", float("nan"))
        row["exp_rmse_C"] = expo.get("rmse", float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    keep = [c for c, _ in COMPARISON_COLUMNS if c in df.columns]
    return df[["experiment"] + [c for c in keep if c != "experiment"]]


def threshold_comparison(processed_by_name: Dict[str, pd.DataFrame], column: str, threshold: float) -> pd.DataFrame:
    """Time for each experiment to first fall to ``threshold`` in ``column`` (Tmean_C or theta_C)."""
    rows = []
    for name, p in processed_by_name.items():
        res = first_downward_crossing(p["elapsed_s"], p[column], threshold)
        rows.append({
            "experiment": name,
            "threshold": threshold,
            "start_value": float(p[column].iloc[0]) if len(p) else float("nan"),
            "end_value": float(p[column].iloc[-1]) if len(p) else float("nan"),
            "measured_range_min": f"{p['elapsed_min'].iloc[0]:.2f} – {p['elapsed_min'].iloc[-1]:.2f}" if len(p) else "—",
            "time_to_threshold_min": res.time_min if res.time_s is not None else float("nan"),
            "status": res.status,
            "result": res.label(),
        })
    return pd.DataFrame(rows)


def common_time_comparison(processed_by_name: Dict[str, pd.DataFrame], target_min: float) -> pd.DataFrame:
    """Interpolated state of each experiment at a common elapsed time (missing if outside its range)."""
    target_s = target_min * 60.0
    rows = []
    for name, p in processed_by_name.items():
        t = p["elapsed_s"]
        inside = len(p) > 0 and t.min() <= target_s <= t.max()
        rows.append({
            "experiment": name,
            "target_time_min": target_min,
            "measured_range_min": f"{p['elapsed_min'].iloc[0]:.2f} – {p['elapsed_min'].iloc[-1]:.2f}" if len(p) else "—",
            "within_range": inside,
            "Tmean_C": value_at_time(t, p["Tmean_C"], target_s),
            "theta_C": value_at_time(t, p["theta_C"], target_s),
            "theta_norm": value_at_time(t, p["theta_norm"], target_s),
            "cumulative_energy_J": value_at_time(t, p["cumulative_energy_J"], target_s),
            "T2_C": value_at_time(t, p["T2_C"], target_s),
            "T3_C": value_at_time(t, p["T3_C"], target_s),
        })
    return pd.DataFrame(rows)
