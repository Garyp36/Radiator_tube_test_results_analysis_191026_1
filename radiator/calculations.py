"""Core engineering calculations for the radiator tube cooling analysis.

Conventions
-----------
* Temperatures in degrees Celsius; temperature *differences* in degrees C are
  numerically identical to kelvin, so cp in J/(kg*K) can be applied directly.
* Elapsed time stored in seconds (``elapsed_s``) with a derived ``elapsed_min``.
* Energy in joules (J), power in watts (W = J/s), heat flux in W/m^2.
* Positive cooling rate / energy removed = water temperature falling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MIN_VALID_DT_S = 1e-9


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSettings:
    """All editable physical inputs for one experiment."""

    water_volume_ml: float = 20.0
    water_density_kg_m3: float = 998.0
    measured_water_mass_kg: Optional[float] = None
    cp_J_kgK: float = 4180.0
    ambient_fallback_C: float = 30.0
    tube_length_mm: float = 100.0
    tube_width_mm: float = 16.0
    tube_height_mm: float = 3.0
    area_method: str = "perimeter"  # "perimeter" or "manual"
    manual_area_m2: Optional[float] = None
    air_velocity_m_s: Optional[float] = None
    design_label: str = ""

    def water_mass_kg(self) -> float:
        return water_mass_kg(self.water_volume_ml, self.water_density_kg_m3, self.measured_water_mass_kg)

    def exposed_area_m2(self) -> Tuple[float, str]:
        return resolve_exposed_area(self.area_method, self.tube_length_mm, self.tube_width_mm,
                                    self.tube_height_mm, self.manual_area_m2)


# --------------------------------------------------------------------------- #
# Water mass and geometry
# --------------------------------------------------------------------------- #
def water_mass_kg(volume_ml: float, density_kg_m3: float, measured_mass_kg: Optional[float] = None) -> float:
    """Total water mass. A directly measured mass (> 0) overrides the volume-based estimate.

    mass [kg] = volume [mL] * 1e-6 [m^3/mL] * density [kg/m^3]
    """
    if measured_mass_kg is not None and np.isfinite(measured_mass_kg) and measured_mass_kg > 0:
        return float(measured_mass_kg)
    if volume_ml is None or density_kg_m3 is None or volume_ml <= 0 or density_kg_m3 <= 0:
        raise ValueError("Water volume and density must be positive numbers.")
    return float(volume_ml) * 1e-6 * float(density_kg_m3)


def rectangular_tube_area_m2(length_mm: float, width_mm: float, height_mm: float) -> float:
    """External area of the four long sides of a rectangular tube (end faces excluded).

    Area [m^2] = 2 * (width + height) * length, with all dimensions converted mm -> m.
    """
    if min(length_mm, width_mm, height_mm) <= 0:
        raise ValueError("Tube dimensions must be positive.")
    length_m, width_m, height_m = length_mm / 1000.0, width_mm / 1000.0, height_mm / 1000.0
    return 2.0 * (width_m + height_m) * length_m


def resolve_exposed_area(method: str, length_mm: float, width_mm: float, height_mm: float,
                         manual_area_m2: Optional[float]) -> Tuple[float, str]:
    """Return (area_m2, description) according to the chosen method."""
    if method == "manual":
        if manual_area_m2 is None or not np.isfinite(manual_area_m2) or manual_area_m2 <= 0:
            raise ValueError("Manual exposed area must be a positive number in m².")
        return float(manual_area_m2), "Manual exposed area entered by user"
    area = rectangular_tube_area_m2(length_mm, width_mm, height_mm)
    return area, f"Rectangular external perimeter: 2 × ({width_mm:g} + {height_mm:g}) mm × {length_mm:g} mm"


# --------------------------------------------------------------------------- #
# Elementary formulas (used both by the table builder and by the tests)
# --------------------------------------------------------------------------- #
def mean_water_temperature(t2: float, t3: float) -> float:
    """Mean of the two endpoint temperatures (an approximation of the bulk water temperature)."""
    return (t2 + t3) / 2.0


def sensible_energy_J(mass_kg: float, cp_J_kgK: float, t_initial_C: float, t_current_C: float) -> float:
    """Signed sensible energy removed: m * cp * (T_initial - T_current). Negative if water warmed."""
    return mass_kg * cp_J_kgK * (t_initial_C - t_current_C)


def heat_loss_rate_W(energy_J: float, duration_s: float) -> float:
    """Average power over an interval. NaN for zero/invalid durations."""
    if duration_s is None or not np.isfinite(duration_s) or duration_s <= MIN_VALID_DT_S or not np.isfinite(energy_J):
        return float("nan")
    return energy_J / duration_s


def heat_flux_W_m2(power_W: float, area_m2: float) -> float:
    """Apparent heat flux = power / area. NaN for invalid inputs."""
    if area_m2 is None or not np.isfinite(area_m2) or area_m2 <= 0 or not np.isfinite(power_W):
        return float("nan")
    return power_W / area_m2


def cooling_rate_C_per_h(t_start_C: float, t_end_C: float, duration_s: float) -> float:
    """Signed cooling rate: -(T_end - T_start) / dt * 3600. Positive = cooling."""
    if duration_s is None or not np.isfinite(duration_s) or duration_s <= MIN_VALID_DT_S:
        return float("nan")
    if not (np.isfinite(t_start_C) and np.isfinite(t_end_C)):
        return float("nan")
    return -(t_end_C - t_start_C) / duration_s * 3600.0


# --------------------------------------------------------------------------- #
# Row selection helpers
# --------------------------------------------------------------------------- #
def apply_time_window(df: pd.DataFrame, start_s: Optional[float], end_s: Optional[float]) -> pd.DataFrame:
    """Keep readings with start_s <= elapsed_s <= end_s (None = open-ended)."""
    mask = pd.Series(True, index=df.index)
    if start_s is not None:
        mask &= df["elapsed_s"] >= start_s
    if end_s is not None:
        mask &= df["elapsed_s"] <= end_s
    return df[mask].reset_index(drop=True)


def reduce_to_report_interval(df: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    """Select the nearest *actual* observation to each target report time.

    Targets are first_time, first_time + interval, ... up to the last time. The first
    and last observations are always retained, duplicates removed, order preserved.
    No synthetic rows are created.
    """
    if df.empty:
        return df.copy()
    if interval_s is None or interval_s <= 0:
        return df.copy()
    times = df["elapsed_s"].to_numpy(dtype=float)
    t_first, t_last = times[0], times[-1]
    targets = np.arange(t_first, t_last + 1e-9, interval_s)
    positions = [int(np.argmin(np.abs(times - target))) for target in targets]
    positions += [0, len(times) - 1]
    unique_positions = sorted(set(positions))
    return df.iloc[unique_positions].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Processed-readings table
# --------------------------------------------------------------------------- #
def build_processed_table(df: pd.DataFrame, settings: ExperimentSettings, experiment_name: str = "experiment",
                          warnings: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute every per-reading and per-interval quantity for the supplied readings.

    The first row of ``df`` is treated as the start of the analysis; interval
    quantities are blank (NaN) for that row while cumulative quantities are 0.
    """
    if warnings is None:
        warnings = []
    out = pd.DataFrame()
    n = len(df)
    out["experiment"] = [experiment_name] * n
    out["elapsed_s"] = df["elapsed_s"].to_numpy(dtype=float)
    out["elapsed_min"] = out["elapsed_s"] / 60.0
    if "datetime" in df.columns:
        out["datetime"] = df["datetime"].to_numpy()
    for col in ("T0_C", "T1_C", "T2_C", "T3_C"):
        out[col] = df[col].to_numpy(dtype=float) if col in df.columns else np.nan

    # Ambient: measured T0 per row where available, otherwise the configured fallback.
    measured = out["T0_C"].to_numpy(dtype=float)
    fallback_mask = ~np.isfinite(measured)
    out["ambient_used_C"] = np.where(fallback_mask, settings.ambient_fallback_C, measured)
    out["ambient_source"] = np.where(fallback_mask, "fallback", "measured T0")
    if fallback_mask.any():
        warnings.append(f"Ambient fallback {settings.ambient_fallback_C:g} °C used for "
                        f"{int(fallback_mask.sum())} of {n} readings (T0 missing).")

    out["Tmean_C"] = mean_water_temperature(out["T2_C"], out["T3_C"])
    out["T_gap_abs_C"] = (out["T2_C"] - out["T3_C"]).abs()
    out["theta_C"] = out["Tmean_C"] - out["ambient_used_C"]

    if n == 0:
        return out

    theta0 = out["theta_C"].iloc[0]
    if np.isfinite(theta0) and abs(theta0) > 1e-12:
        out["theta_norm"] = out["theta_C"] / theta0
    else:
        out["theta_norm"] = np.nan
        warnings.append("Initial temperature excess θ₀ is zero or invalid; normalised excess cannot be computed.")

    mass = settings.water_mass_kg()
    cp = settings.cp_J_kgK
    try:
        area, _ = settings.exposed_area_m2()
    except ValueError as exc:
        area = float("nan")
        warnings.append(str(exc))

    dt = np.diff(out["elapsed_s"].to_numpy(dtype=float), prepend=np.nan)
    dt_valid = np.where(np.isfinite(dt) & (dt > MIN_VALID_DT_S), dt, np.nan)
    out["interval_dt_s"] = dt_valid
    if n > 1 and np.isnan(dt_valid[1:]).any():
        warnings.append("One or more intervals have zero duration; interval rates for those rows are blank.")

    for col, rate_col in (("T2_C", "rate_T2_C_per_h"), ("T3_C", "rate_T3_C_per_h"), ("Tmean_C", "rate_Tmean_C_per_h")):
        values = out[col].to_numpy(dtype=float)
        dT = np.diff(values, prepend=np.nan)
        out[rate_col] = -dT / dt_valid * 3600.0

    tmean = out["Tmean_C"].to_numpy(dtype=float)
    d_tmean = np.diff(tmean, prepend=np.nan)
    out["interval_energy_J"] = mass * cp * (-d_tmean)
    out["interval_heat_loss_W"] = out["interval_energy_J"] / dt_valid
    out["interval_heat_flux_W_m2"] = out["interval_heat_loss_W"] / area if np.isfinite(area) and area > 0 else np.nan

    out["cumulative_energy_J"] = mass * cp * (tmean[0] - tmean)
    elapsed_from_start = out["elapsed_s"] - out["elapsed_s"].iloc[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_power = np.where(elapsed_from_start > MIN_VALID_DT_S,
                             out["cumulative_energy_J"] / elapsed_from_start, np.nan)
    out["cumulative_avg_heat_loss_W"] = cum_power
    out["cumulative_avg_heat_flux_W_m2"] = cum_power / area if np.isfinite(area) and area > 0 else np.nan

    # Carry operating-condition and status columns through untouched.
    for col in df.columns:
        if col not in out.columns and col not in ("elapsed_s",):
            out[col] = df[col].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarise_experiment(processed: pd.DataFrame, settings: ExperimentSettings,
                         experiment_name: str = "experiment") -> Dict[str, object]:
    """Window-level results derived from a processed table (first row -> last row)."""
    s: Dict[str, object] = {"experiment": experiment_name, "design": settings.design_label or experiment_name}
    n = len(processed)
    s["n_readings"] = n
    mass = settings.water_mass_kg()
    s["water_mass_kg"] = mass
    s["cp_J_kgK"] = settings.cp_J_kgK
    try:
        area, area_method = settings.exposed_area_m2()
    except ValueError as exc:
        area, area_method = float("nan"), str(exc)
    s["exposed_area_m2"] = area
    s["area_method"] = area_method
    s["air_velocity_m_s"] = settings.air_velocity_m_s
    if n < 2:
        for key in ("duration_s", "duration_min", "T2_initial_C", "T2_final_C", "T3_initial_C", "T3_final_C",
                    "Tmean_initial_C", "Tmean_final_C", "Tmean_drop_C", "rate_T2_C_per_h", "rate_T3_C_per_h",
                    "rate_Tmean_C_per_h", "gap_mean_C", "gap_max_C", "energy_removed_J", "avg_heat_loss_W",
                    "apparent_heat_flux_W_m2", "ambient_initial_C", "ambient_final_C", "theta_initial_C",
                    "theta_final_C"):
            s[key] = float("nan")
        s["ambient_source"] = "—"
        return s

    first, last = processed.iloc[0], processed.iloc[-1]
    duration_s = float(last["elapsed_s"] - first["elapsed_s"])
    s["duration_s"] = duration_s
    s["duration_min"] = duration_s / 60.0
    for col, key in (("T2_C", "T2"), ("T3_C", "T3"), ("Tmean_C", "Tmean")):
        s[f"{key}_initial_C"] = float(first[col])
        s[f"{key}_final_C"] = float(last[col])
        s[f"rate_{key}_C_per_h"] = cooling_rate_C_per_h(float(first[col]), float(last[col]), duration_s)
    s["Tmean_drop_C"] = float(first["Tmean_C"] - last["Tmean_C"])
    s["gap_mean_C"] = float(processed["T_gap_abs_C"].mean())
    s["gap_max_C"] = float(processed["T_gap_abs_C"].max())
    energy = sensible_energy_J(mass, settings.cp_J_kgK, float(first["Tmean_C"]), float(last["Tmean_C"]))
    s["energy_removed_J"] = energy
    s["avg_heat_loss_W"] = heat_loss_rate_W(energy, duration_s)
    s["apparent_heat_flux_W_m2"] = heat_flux_W_m2(s["avg_heat_loss_W"], area)
    s["ambient_initial_C"] = float(first["ambient_used_C"])
    s["ambient_final_C"] = float(last["ambient_used_C"])
    sources = set(processed["ambient_source"].unique())
    s["ambient_source"] = "measured T0" if sources == {"measured T0"} else (
        "fallback" if sources == {"fallback"} else "mixed (measured + fallback)")
    s["theta_initial_C"] = float(first["theta_C"])
    s["theta_final_C"] = float(last["theta_C"])
    for col in ("fan_pwm_pct", "fan_rpm"):
        s[col] = float(processed[col].median()) if col in processed.columns and processed[col].notna().any() else None
    return s


# --------------------------------------------------------------------------- #
# Threshold crossing and common-time interpolation
# --------------------------------------------------------------------------- #
@dataclass
class CrossingResult:
    time_s: Optional[float]
    status: str  # "crossed", "already_below_at_start", "not_reached", "insufficient_data"

    @property
    def time_min(self) -> Optional[float]:
        return None if self.time_s is None else self.time_s / 60.0

    def label(self) -> str:
        if self.status == "crossed":
            return f"{self.time_min:.2f} min"
        if self.status == "already_below_at_start":
            return "0.00 min (already below threshold at start)"
        if self.status == "not_reached":
            return "Not reached"
        return "Insufficient data"


def first_downward_crossing(times_s, values, threshold: float) -> CrossingResult:
    """Time of the first downward crossing of ``threshold`` within recorded data.

    * Linear interpolation between the bracketing readings; never extrapolates.
    * If the first valid value is already <= threshold -> 0 s with status flag.
    * If the data never drop to the threshold -> ``not_reached``.
    """
    t = np.asarray(times_s, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) == 0:
        return CrossingResult(None, "insufficient_data")
    if v[0] <= threshold:
        return CrossingResult(0.0, "already_below_at_start")
    for i in range(1, len(t)):
        if v[i] <= threshold < v[i - 1]:
            if v[i] == threshold or v[i - 1] == v[i]:
                return CrossingResult(float(t[i] - t[0]), "crossed")
            frac = (v[i - 1] - threshold) / (v[i - 1] - v[i])
            return CrossingResult(float(t[i - 1] + frac * (t[i] - t[i - 1]) - t[0]), "crossed")
    return CrossingResult(None, "not_reached")


def value_at_time(times_s, values, target_s: float) -> float:
    """Linearly interpolated value at ``target_s``; NaN when outside the measured interval."""
    t = np.asarray(times_s, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) == 0 or target_s < t.min() - 1e-9 or target_s > t.max() + 1e-9:
        return float("nan")
    order = np.argsort(t)
    return float(np.interp(target_s, t[order], v[order]))


def format_value(value, fmt: str = "{:.3f}", missing: str = "—") -> str:
    """Human-readable formatting with an explicit marker for unavailable values."""
    if value is None:
        return missing
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return missing
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)
