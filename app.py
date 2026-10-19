"""Radiator Tube Cooling Analysis - Streamlit application.

Run with:  streamlit run app.py
All numbers displayed are produced by the functions in the ``radiator`` package
from the uploaded data; nothing is hard-coded sample output.
"""
from __future__ import annotations

import io
import os
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from radiator import __version__
from radiator.calculations import (ExperimentSettings, apply_time_window, build_processed_table, format_value,
                                   reduce_to_report_interval, summarise_experiment)
from radiator.comparison import (COMPARISON_COLUMNS, build_comparison_table, common_time_comparison,
                                 threshold_comparison)
from radiator.data_io import COLUMN_ALIASES, DUPLICATE_POLICY, LoadResult, load_experiment_csv, metadata_from_data
from radiator.fitting import fit_exponential, fit_linear
from radiator.plots import comparison_plot, energy_plot, rates_plot, temperature_plot, theta_plot
from radiator.storage import SAVED_RUNS_DIR, list_saved_runs, save_run

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILES = ["sample_experiment.csv", "sample_experiment_b.csv"]

st.set_page_config(page_title="Radiator Tube Cooling Analysis", page_icon="🌡️", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def optional_positive(value: float) -> Optional[float]:
    """Number inputs cannot be blank; 0 is used to mean 'not provided'."""
    return float(value) if value and value > 0 else None


def settings_form(prefix: str, defaults: ExperimentSettings, csv_air_velocity: Optional[float] = None,
                  show_design_label: bool = True) -> ExperimentSettings:
    """Render the editable physical inputs and return an ExperimentSettings."""
    design = defaults.design_label
    if show_design_label:
        design = st.text_input("Experiment name / tube design label", value=defaults.design_label,
                               key=f"{prefix}_design")
    st.markdown("**Water properties**")
    c1, c2 = st.columns(2)
    volume = c1.number_input("Total water volume (mL) – tube + silicone passages", min_value=0.01,
                             value=float(defaults.water_volume_ml), step=1.0, key=f"{prefix}_vol")
    density = c2.number_input("Water density (kg/m³)", min_value=1.0, value=float(defaults.water_density_kg_m3),
                              step=1.0, key=f"{prefix}_rho")
    measured_mass = c1.number_input("Measured water mass (kg) – 0 = not measured, overrides volume when > 0",
                                    min_value=0.0, value=float(defaults.measured_water_mass_kg or 0.0),
                                    step=0.001, format="%.4f", key=f"{prefix}_mass")
    cp = c2.number_input("Specific heat cp (J/(kg·K))", min_value=1.0, value=float(defaults.cp_J_kgK), step=10.0,
                         key=f"{prefix}_cp")
    st.markdown("**Ambient conditions**")
    ambient = st.number_input("Constant ambient fallback (°C) – used only for readings without T0",
                              value=float(defaults.ambient_fallback_C), step=0.5, key=f"{prefix}_amb")
    st.markdown("**Tube geometry**")
    g1, g2, g3 = st.columns(3)
    length = g1.number_input("Tube length (mm)", min_value=0.01, value=float(defaults.tube_length_mm), step=1.0,
                             key=f"{prefix}_len")
    width = g2.number_input("External width (mm)", min_value=0.01, value=float(defaults.tube_width_mm), step=0.5,
                            key=f"{prefix}_wid")
    height = g3.number_input("External height (mm)", min_value=0.01, value=float(defaults.tube_height_mm),
                             step=0.5, key=f"{prefix}_hei")
    method_label = st.radio("Heat-flux area method", ["Rectangular external perimeter × length", "Manual exposed area"],
                            index=0 if defaults.area_method == "perimeter" else 1, horizontal=True,
                            key=f"{prefix}_areamethod")
    method = "perimeter" if method_label.startswith("Rectangular") else "manual"
    manual_area = defaults.manual_area_m2
    if method == "manual":
        manual_area = st.number_input("Exposed area (m²)", min_value=0.0, value=float(defaults.manual_area_m2 or 0.0),
                                      step=0.0001, format="%.6f", key=f"{prefix}_area")
    st.markdown("**Air flow**")
    if csv_air_velocity is not None:
        st.caption(f"Air velocity detected in CSV (median): {csv_air_velocity:.3g} m/s. "
                   "Enter 0 below to use the CSV value, or type a value to override it.")
        air_default = 0.0
    else:
        st.caption("No air-velocity column in CSV; enter the metadata value.")
        air_default = float(defaults.air_velocity_m_s or 0.0)
    air = st.number_input("Air velocity (m/s)", min_value=0.0, value=air_default, step=0.1, key=f"{prefix}_air")
    air_velocity = optional_positive(air) if optional_positive(air) is not None else csv_air_velocity
    return ExperimentSettings(
        water_volume_ml=volume, water_density_kg_m3=density, measured_water_mass_kg=optional_positive(measured_mass),
        cp_J_kgK=cp, ambient_fallback_C=ambient, tube_length_mm=length, tube_width_mm=width, tube_height_mm=height,
        area_method=method, manual_area_m2=optional_positive(manual_area) if manual_area else None,
        air_velocity_m_s=air_velocity, design_label=design,
    )


def analyse(load: LoadResult, settings: ExperimentSettings, opts: Dict[str, object]) -> Dict[str, object]:
    """Window -> optional interval reduction -> processed table -> summary -> fits."""
    warnings: List[str] = list(load.warnings)
    windowed = apply_time_window(load.data, opts["window_start_s"], opts["window_end_s"])
    if len(windowed) < 2:
        return {"ok": False, "warnings": warnings + ["Fewer than two readings inside the selected time window."]}
    full = build_processed_table(windowed, settings, load.name, warnings)
    reduced = None
    if opts["use_interval"]:
        reduced_rows = reduce_to_report_interval(windowed, opts["interval_s"])
        reduced = build_processed_table(reduced_rows, settings, load.name, [])
    basis_label = "all valid readings"
    basis = full
    if opts["use_interval"] and opts["basis"] == "reduced" and reduced is not None:
        basis = reduced
        basis_label = f"interval-reduced readings (nearest observation every {opts['interval_s']:g} s)"
    summary = summarise_experiment(basis, settings, load.name)
    fits = {"linear": fit_linear(basis["elapsed_min"], basis["Tmean_C"]),
            "exponential": fit_exponential(basis["elapsed_min"], basis["theta_C"])}
    return {"ok": True, "load": load, "settings": settings, "full": full, "reduced": reduced, "basis": basis,
            "basis_label": basis_label, "summary": summary, "fits": fits, "warnings": warnings}


def summary_rows(s: Dict[str, object]) -> pd.DataFrame:
    f = format_value
    rows = [
        ("Analysis duration", f"{f(s['duration_min'], '{:.2f}')} min ({f(s['duration_s'], '{:.0f}')} s)"),
        ("Number of readings used", str(s["n_readings"])),
        ("Initial / final T2 (inlet end)", f"{f(s['T2_initial_C'], '{:.2f}')} / {f(s['T2_final_C'], '{:.2f}')} °C"),
        ("Initial / final T3 (outlet end)", f"{f(s['T3_initial_C'], '{:.2f}')} / {f(s['T3_final_C'], '{:.2f}')} °C"),
        ("Initial / final Tmean (endpoint mean)", f"{f(s['Tmean_initial_C'], '{:.2f}')} / {f(s['Tmean_final_C'], '{:.2f}')} °C"),
        ("Mean water temperature drop", f"{f(s['Tmean_drop_C'], '{:.2f}')} °C"),
        ("T2 average cooling rate (endpoint-to-endpoint)", f"{f(s['rate_T2_C_per_h'], '{:.3f}')} °C/h"),
        ("T3 average cooling rate (endpoint-to-endpoint)", f"{f(s['rate_T3_C_per_h'], '{:.3f}')} °C/h"),
        ("Tmean average cooling rate (endpoint-to-endpoint)", f"{f(s['rate_Tmean_C_per_h'], '{:.3f}')} °C/h"),
        ("Mean / maximum |T2 − T3|", f"{f(s['gap_mean_C'], '{:.2f}')} / {f(s['gap_max_C'], '{:.2f}')} °C"),
        ("Water mass used", f"{f(s['water_mass_kg'], '{:.5f}')} kg"),
        ("Specific heat cp used", f"{f(s['cp_J_kgK'], '{:.0f}')} J/(kg·K)"),
        ("Sensible energy removed", f"{f(s['energy_removed_J'], '{:.1f}')} J"),
        ("Average net heat-loss rate", f"{f(s['avg_heat_loss_W'], '{:.4f}')} W"),
        ("Exposed surface area", f"{f(s['exposed_area_m2'], '{:.6f}')} m² – {s['area_method']}"),
        ("Apparent average heat flux (estimated)", f"{f(s['apparent_heat_flux_W_m2'], '{:.2f}')} W/m²"),
        ("Ambient temperature at start / end", f"{f(s['ambient_initial_C'], '{:.2f}')} / {f(s['ambient_final_C'], '{:.2f}')} °C ({s['ambient_source']})"),
        ("Temperature excess θ at start / end", f"{f(s['theta_initial_C'], '{:.2f}')} / {f(s['theta_final_C'], '{:.2f}')} °C"),
        ("Air velocity", f"{f(s['air_velocity_m_s'], '{:.2f}')} m/s"),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def summary_frame(summaries: List[Dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(summaries)


def fits_frame(name: str, fits: Dict[str, Dict[str, object]], basis_label: str) -> pd.DataFrame:
    rows = []
    for model, fit in fits.items():
        row = {"experiment": name, "model": model, "fit_ok": fit.get("ok"), "message": fit.get("message", ""),
               "basis": basis_label, "n_points": fit.get("n_points"), "equation": fit.get("equation"),
               "r2": fit.get("r2"), "rmse_C": fit.get("rmse")}
        for key in ("intercept_C", "slope_C_per_min", "cooling_rate_C_per_h", "A_C", "k_per_min", "tau_min"):
            row[key] = fit.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def residuals_frame(name: str, fits: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    frames = []
    for model, fit in fits.items():
        if fit.get("ok") and fit.get("fitted") is not None:
            df = fit["fitted"].copy()
            df.insert(0, "model", model)
            df.insert(0, "experiment", name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def show_fit_section(fits: Dict[str, Dict[str, object]], basis_label: str) -> None:
    st.subheader("Curve fitting and statistics")
    st.caption(f"Fits use **{basis_label}**. Fit quality (R², RMSE) describes how well a curve follows the data; "
               "it does not by itself prove that the model is physically correct.")
    c1, c2 = st.columns(2)
    lin, expo = fits["linear"], fits["exponential"]
    with c1:
        st.markdown("**Linear model:** `Tmean = intercept + slope × t`")
        if lin["ok"]:
            st.code(lin["equation"])
            st.table(pd.DataFrame({
                "Parameter": ["Intercept (°C)", "Slope (°C/min)", "Equivalent cooling rate (°C/h)", "R²", "RMSE (°C)", "Points"],
                "Value": [format_value(lin["intercept_C"], "{:.4f}"), format_value(lin["slope_C_per_min"], "{:.5f}"),
                          format_value(lin["cooling_rate_C_per_h"], "{:.3f}"), format_value(lin["r2"], "{:.4f}"),
                          format_value(lin["rmse"], "{:.3f}"), str(lin["n_points"])]}))
        else:
            st.warning(f"Linear fit unavailable: {lin['message']}")
    with c2:
        st.markdown("**Exponential cooling model:** `θ(t) = A × exp(−k × t)`, θ = Tmean − T0")
        if expo["ok"]:
            st.code(expo["equation"])
            st.table(pd.DataFrame({
                "Parameter": ["A (°C)", "k (min⁻¹)", "Time constant τ = 1/k (min)", "R²", "RMSE (°C)", "Points"],
                "Value": [format_value(expo["A_C"], "{:.4f}"), format_value(expo["k_per_min"], "{:.5f}"),
                          format_value(expo["tau_min"], "{:.2f}") + ("" if expo["k_per_min"] > 0 else " (k ≤ 0: not defined)"),
                          format_value(expo["r2"], "{:.4f}"), format_value(expo["rmse"], "{:.3f}"), str(expo["n_points"])]}))
        else:
            st.warning(f"Exponential fit unavailable: {expo['message']}")
    with st.expander("Predicted values and residuals"):
        for model, fit in fits.items():
            if fit["ok"]:
                st.markdown(f"*{model.capitalize()} model*")
                st.dataframe(fit["fitted"], width="stretch", hide_index=True)


def show_detection(load: LoadResult) -> None:
    essential = ["elapsed_s", "T2_C", "T3_C"]
    detected = [c for c in load.detected if c in essential] + (["datetime"] if "datetime" in load.detected else [])
    optional_found = [c for c in load.detected if c not in essential and c != "datetime"]
    if load.missing_required:
        st.error(f"**{load.name}** – missing required column(s): {', '.join(load.missing_required)}. "
                 f"Detected: {', '.join(load.detected) or 'none'}.")
    else:
        st.success(f"**{load.name}** – required columns detected: {', '.join(detected)}. "
                   f"Optional columns found: {', '.join(optional_found) or 'none'}. "
                   f"Missing optional: {', '.join(load.missing_optional) or 'none'}. "
                   f"Valid readings: {len(load.data)} of {len(load.raw)}.")
    for w in load.warnings:
        st.warning(f"{load.name}: {w}")


def processed_display(p: pd.DataFrame) -> pd.DataFrame:
    """Order columns for display; NaN shown blank by Streamlit."""
    preferred = ["experiment", "elapsed_s", "elapsed_min", "T0_C", "T1_C", "T2_C", "T3_C", "ambient_used_C",
                 "ambient_source", "Tmean_C", "T_gap_abs_C", "theta_C", "theta_norm", "rate_T2_C_per_h",
                 "rate_T3_C_per_h", "rate_Tmean_C_per_h", "interval_dt_s", "interval_energy_J",
                 "interval_heat_loss_W", "interval_heat_flux_W_m2", "cumulative_energy_J",
                 "cumulative_avg_heat_loss_W", "cumulative_avg_heat_flux_W_m2"]
    cols = [c for c in preferred if c in p.columns] + [c for c in p.columns if c not in preferred]
    return p[cols]


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Radiator Tube Cooling Analysis")
st.markdown("*Heat Transfer | Heat Flux | Cooling Rates | Curve Fitting | Experiment Comparison*")
st.caption(f"Version {__version__}. Stationary-water cooling test: T2 and T3 are **endpoint** water temperatures; "
           "their arithmetic mean Tmean is an *approximation* of the representative water temperature. "
           "The app estimates sensible-energy change of the water; it does **not** measure a steady-state radiator "
           "heat-transfer coefficient or a direct tube-wall heat flux.")

with st.expander("What is calculated?"):
    st.markdown(f"""
| Quantity | Formula | Units |
|---|---|---|
| Mean water temperature (endpoint mean) | `Tmean = (T2 + T3) / 2` | °C |
| Endpoint gap | `abs(T2 − T3)` (mean and max over window) | °C |
| Water mass | `m = V[mL] × 1e-6 × ρ[kg/m³]` (or measured mass) | kg |
| Interval cooling rate | `−ΔT / Δt × 3600` (signed; negative if warming) | °C/h |
| Cumulative sensible energy removed | `m × cp × (Tmean,initial − Tmean,current)` | J |
| Interval energy removed | `m × cp × (Tmean,prev − Tmean,current)` | J |
| Interval net heat-loss rate | `interval energy / interval duration` | W |
| Average net heat-loss rate | `total energy / total duration` | W |
| Exposed area (rectangular tube) | `2 × (w + h) × L` (mm → m) | m² |
| Apparent heat flux | `heat-loss rate / exposed area` | W/m² |
| Temperature excess | `θ = Tmean − T0` ; normalised `θ / θ₀` | °C, – |
| Linear fit | `Tmean = a + b·t` | – |
| Exponential fit | `θ = A·exp(−k·t)`, τ = 1/k | – |

Energy (J) is a quantity of heat; power (W = J/s) is its rate; heat flux (W/m²) is power per unit area.
Accepted column aliases: {"; ".join(f"**{k}**: {', '.join(v)}" for k, v in COLUMN_ALIASES.items())}.
Duplicate-time policy: {DUPLICATE_POLICY}
""")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    mode = st.radio("Analysis mode", ["Single Experiment Analysis", "Experiment Comparison"])
    uploads = st.file_uploader("Upload experiment CSV file(s)", type=["csv"], accept_multiple_files=True)
    use_sample = st.checkbox("Use bundled sample data (not a real experimental result)", value=not uploads)

    st.header("Common experiment settings")
    st.caption("Defaults for every experiment. Each experiment can override these in the main panel.")
    common = settings_form("common", ExperimentSettings(air_velocity_m_s=1.1), show_design_label=False)

    st.header("Analysis settings")
    use_interval = st.checkbox("Enable report interval (nearest actual observations)", value=False)
    interval_s = st.number_input("Report interval (s)", min_value=1.0, value=600.0, step=60.0, disabled=not use_interval)
    basis_choice = st.radio("Readings used for summary metrics and curve fits",
                            ["All valid readings", "Nearest observations at report interval"],
                            index=0, disabled=not use_interval)
    use_window = st.checkbox("Restrict analysis to a time window", value=False)
    w1, w2 = st.columns(2)
    window_start_min = w1.number_input("Window start (min)", min_value=0.0, value=0.0, step=1.0, disabled=not use_window)
    window_end_min = w2.number_input("Window end (min)", min_value=0.0, value=90.0, step=1.0, disabled=not use_window)
    st.markdown("**Curve fitting**")
    show_linear = st.checkbox("Overlay linear fit on temperature plot", value=True)
    show_exp = st.checkbox("Overlay exponential fit", value=True)

    opts = {
        "use_interval": use_interval, "interval_s": float(interval_s),
        "basis": "reduced" if (use_interval and basis_choice.startswith("Nearest")) else "all",
        "window_start_s": window_start_min * 60.0 if use_window else None,
        "window_end_s": window_end_min * 60.0 if use_window else None,
    }
    if use_window and window_end_min <= window_start_min:
        st.error("Window end must be later than window start.")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
loads: List[LoadResult] = []
if uploads:
    for up in uploads:
        loads.append(load_experiment_csv(io.BytesIO(up.getvalue()), name=up.name))
if use_sample:
    for fname in SAMPLE_FILES:
        path = os.path.join(APP_DIR, fname)
        if os.path.exists(path):
            loads.append(load_experiment_csv(path, name=f"{fname} (sample)"))

if not loads:
    st.info("Upload one or more CSV files in the sidebar, or tick *Use bundled sample data* to explore the app.")
    st.stop()

st.subheader("Column detection and data quality")
for load in loads:
    show_detection(load)
usable = [l for l in loads if l.ok]
if not usable:
    st.error("No uploaded file has the essential columns (elapsed time, T2, T3) with at least two valid readings.")
    st.stop()


def experiment_settings_panel(load: LoadResult, idx: int) -> ExperimentSettings:
    meta = metadata_from_data(load.data)
    csv_air = meta.get("air_speed_m_s")
    with st.expander(f"Settings for **{load.name}** (override common settings here)", expanded=False):
        override = st.checkbox("Override common geometry / water / air-velocity settings for this experiment",
                               value=False, key=f"ovr_{idx}")
        if override:
            defaults = ExperimentSettings(**{**asdict(common), "design_label": load.name})
            return settings_form(f"exp{idx}", defaults, csv_air_velocity=csv_air)
        design = st.text_input("Experiment name / tube design label", value=load.name, key=f"design_{idx}")
        settings = ExperimentSettings(**{**asdict(common), "design_label": design})
        if csv_air is not None:
            settings.air_velocity_m_s = csv_air
            st.caption(f"Air velocity taken from CSV (median): {csv_air:.3g} m/s. Geometry and water properties from common settings.")
        else:
            st.caption(f"Using common settings; air velocity {common.air_velocity_m_s} m/s (metadata, no CSV column).")
        return settings


# --------------------------------------------------------------------------- #
# Single experiment analysis
# --------------------------------------------------------------------------- #
if mode == "Single Experiment Analysis":
    names = [l.name for l in usable]
    chosen = st.selectbox("Experiment to analyse", names) if len(names) > 1 else names[0]
    load = usable[names.index(chosen)]
    settings = experiment_settings_panel(load, 0)
    result = analyse(load, settings, opts)
    for w in result["warnings"]:
        if w not in load.warnings:
            st.warning(w)
    if not result["ok"]:
        st.stop()
    s, p, fits = result["summary"], result["basis"], result["fits"]

    st.subheader("Key results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensible energy removed", f"{format_value(s['energy_removed_J'], '{:.1f}')} J")
    m2.metric("Overall average net heat-loss rate", f"{format_value(s['avg_heat_loss_W'], '{:.4f}')} W")
    m3.metric("Exposed tube area", f"{format_value(s['exposed_area_m2'], '{:.5f}')} m²")
    m4.metric("Apparent average heat flux (estimated)", f"{format_value(s['apparent_heat_flux_W_m2'], '{:.2f}')} W/m²")
    st.caption(f"Based on **{result['basis_label']}**. Heat flux is an *apparent* value derived from the water's "
               "sensible-energy change divided by the geometric exposed area – not a measured wall heat flux. "
               "Ambient source: " + s["ambient_source"] + ".")
    if s["ambient_source"] != "measured T0":
        st.warning(f"Constant ambient fallback ({settings.ambient_fallback_C:g} °C) is being used for some or all readings.")

    st.subheader("Calculation summary")
    st.table(summary_rows(s))
    st.info("Limitations: the energy balance covers only the water's sensible heat. Heat stored in the aluminium tube "
            "and silicone passages, exchange with the heated vessel or surroundings, radiation and sensor dynamics are "
            "not separated out. The geometric area assumes all four long external faces are exposed to air.")

    st.subheader("Plots")
    st.plotly_chart(temperature_plot(p, fits["linear"] if show_linear else None, fits["exponential"] if show_exp else None),
                    width="stretch")
    st.plotly_chart(theta_plot(p, fits["exponential"] if show_exp else None), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(energy_plot(p), width="stretch")
    c2.plotly_chart(rates_plot(p), width="stretch")

    show_fit_section(fits, result["basis_label"])

    st.subheader("Processed data")
    with st.expander("Processed readings (analysis basis)", expanded=False):
        st.caption("Blank cells = not available (e.g. no previous interval for the first row). "
                   "Interval quantities refer to the interval ending at that row.")
        st.dataframe(processed_display(p), width="stretch", hide_index=True)
    if result["reduced"] is not None:
        with st.expander(f"Interval-reduced report table ({opts['interval_s']:g} s, nearest actual observations)"):
            st.dataframe(processed_display(result["reduced"]), width="stretch", hide_index=True)
    with st.expander("Original cleaned readings (as uploaded, after validation)"):
        st.dataframe(load.data, width="stretch", hide_index=True)

    st.subheader("Downloads")
    summary_df = summary_frame([s])
    fit_df = fits_frame(load.name, fits, result["basis_label"])
    resid_df = residuals_frame(load.name, fits)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("Summary CSV", csv_bytes(summary_df), "summary_results.csv", "text/csv")
    d2.download_button("Processed readings (full resolution) CSV", csv_bytes(result["full"]), "processed_readings_full.csv", "text/csv")
    d3.download_button("Interval report table CSV", csv_bytes(result["reduced"]) if result["reduced"] is not None else b"",
                       "processed_readings_interval.csv", "text/csv", disabled=result["reduced"] is None)
    d4.download_button("Fit coefficients CSV", csv_bytes(fit_df), "fit_coefficients.csv", "text/csv")
    d5.download_button("Fit residuals CSV", csv_bytes(resid_df), "fit_residuals.csv", "text/csv", disabled=resid_df.empty)

    st.subheader("Save to local project storage")
    st.caption(f"Files are written to `{SAVED_RUNS_DIR}`. On Streamlit Community Cloud this storage can be "
               "ephemeral and is lost on redeploy/restart. No authentication is implemented: do not deploy this app "
               "publicly with confidential data.")
    label = st.text_input("Run label", value=os.path.splitext(load.name)[0])
    if st.button("Save summary, processed readings and settings"):
        written = save_run(label, summary_df, result["full"],
                           {"experiment_settings": asdict(settings), "analysis_options": opts},
                           [load.name], extra_tables={"fits": fit_df, "residuals": resid_df})
        st.success("Saved: " + ", ".join(f"{w['file']} ({w['bytes']} bytes)" for w in written))
    with st.expander("Saved files"):
        st.dataframe(list_saved_runs(), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Experiment comparison
# --------------------------------------------------------------------------- #
else:
    if len(usable) < 2:
        st.warning("Upload at least two valid experiment files for a comparison (or tick the sample-data box).")
    st.subheader("Experiment-specific settings")
    st.caption("Different tube designs may need different geometry, water volume or air velocity – set them per experiment.")
    results: Dict[str, Dict[str, object]] = {}
    for idx, load in enumerate(usable):
        settings = experiment_settings_panel(load, idx)
        res = analyse(load, settings, opts)
        for w in res["warnings"]:
            if w not in load.warnings:
                st.warning(f"{load.name}: {w}")
        if res["ok"]:
            results[load.name] = res
    if len(results) < 1:
        st.stop()

    with st.sidebar:
        st.header("Comparison controls")
        selected = st.multiselect("Experiments shown in plots", list(results), default=list(results))
        criterion = st.radio("Common comparison criterion", [
            "Time to a common mean water temperature",
            "Time to a common ambient-excess temperature",
            "Performance at a common elapsed time"])
        tmean_threshold = st.number_input("Tmean threshold (°C)", value=50.0, step=0.5)
        theta_threshold = st.number_input("θ threshold (°C)", value=15.0, step=0.5)
        common_time_min = st.number_input("Common elapsed time (min)", min_value=0.0, value=30.0, step=1.0)

    summaries = [r["summary"] for r in results.values()]
    fits_by = {n: r["fits"] for n, r in results.items()}
    processed_by = {n: r["basis"] for n, r in results.items()}

    st.subheader("Comparison summary")
    st.caption("Basis: " + "; ".join(f"{n}: {r['basis_label']}" for n, r in results.items()))
    comp = build_comparison_table(summaries, fits_by)
    rename = dict(COMPARISON_COLUMNS)
    display = comp.rename(columns=rename).set_index("experiment").T
    display = display.apply(lambda col: col.map(lambda v: format_value(v, "{:.4g}") if isinstance(v, (int, float, np.floating)) else ("—" if v is None else str(v))))
    st.dataframe(display, width="stretch")

    st.subheader("Common comparison criterion")
    if criterion.startswith("Time to a common mean"):
        crit_df = threshold_comparison(processed_by, "Tmean_C", tmean_threshold)
        st.markdown(f"Time for Tmean to first fall to **{tmean_threshold:g} °C** (linear interpolation between recorded "
                    "readings; no extrapolation).")
    elif criterion.startswith("Time to a common ambient"):
        crit_df = threshold_comparison(processed_by, "theta_C", theta_threshold)
        st.markdown(f"Time for θ = Tmean − T0 to first fall to **{theta_threshold:g} °C** (ambient-corrected, so "
                    "experiments at different ambient temperatures can be compared).")
    else:
        crit_df = common_time_comparison(processed_by, common_time_min)
        st.markdown(f"State of each experiment at **{common_time_min:g} min** (interpolated only within each "
                    "experiment's measured range; blank otherwise).")
    st.dataframe(crit_df, width="stretch", hide_index=True)

    st.subheader("Overlay plots")
    shown = {n: processed_by[n] for n in selected if n in processed_by}
    if shown:
        st.plotly_chart(comparison_plot(shown, [("T2_C", "T2"), ("T3_C", "T3"), ("Tmean_C", "Tmean")],
                                        "Temperature (°C)", "T2 / T3 / Tmean versus elapsed time"), width="stretch")
        st.plotly_chart(comparison_plot(shown, [("theta_C", "θ = Tmean − T0")], "Temperature excess (°C)",
                                        "Ambient-corrected temperature excess"), width="stretch")
        st.plotly_chart(comparison_plot(shown, [("cumulative_energy_J", "cumulative energy")], "Energy (J)",
                                        "Cumulative sensible energy removed"), width="stretch")
    else:
        st.info("Select at least one experiment in the sidebar to show overlay plots.")

    st.subheader("Curve fits per experiment")
    for n, r in results.items():
        with st.expander(f"Fits – {n}"):
            show_fit_section(r["fits"], r["basis_label"])

    st.subheader("Processed data")
    all_processed = pd.concat([r["full"] for r in results.values()], ignore_index=True)
    with st.expander("Processed readings for all experiments (full resolution)"):
        st.dataframe(processed_display(all_processed), width="stretch", hide_index=True)

    st.subheader("Downloads")
    summary_df = summary_frame(summaries)
    fit_df = pd.concat([fits_frame(n, r["fits"], r["basis_label"]) for n, r in results.items()], ignore_index=True)
    resid_df = pd.concat([residuals_frame(n, r["fits"]) for n, r in results.items()], ignore_index=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("Comparison table CSV", csv_bytes(comp), "comparison_results.csv", "text/csv")
    d2.download_button("Criterion results CSV", csv_bytes(crit_df), "comparison_criterion.csv", "text/csv")
    d3.download_button("Summary CSV", csv_bytes(summary_df), "summary_results.csv", "text/csv")
    d4.download_button("Processed readings CSV", csv_bytes(all_processed), "processed_readings_all.csv", "text/csv")
    d5.download_button("Fit coefficients CSV", csv_bytes(fit_df), "fit_coefficients.csv", "text/csv")

    st.subheader("Save to local project storage")
    st.caption(f"Files are written to `{SAVED_RUNS_DIR}`. Hosted storage (Streamlit Community Cloud) can be ephemeral. "
               "No authentication: do not deploy publicly with confidential data.")
    label = st.text_input("Run label", value="comparison")
    if st.button("Save comparison, processed readings and settings"):
        written = save_run(label, summary_df, all_processed,
                           {"experiment_settings": {n: asdict(r["settings"]) for n, r in results.items()},
                            "analysis_options": opts, "criterion": criterion},
                           list(results), extra_tables={"comparison": comp, "criterion": crit_df, "fits": fit_df,
                                                         "residuals": resid_df})
        st.success("Saved: " + ", ".join(f"{w['file']} ({w['bytes']} bytes)" for w in written))
    with st.expander("Saved files"):
        st.dataframe(list_saved_runs(), width="stretch", hide_index=True)
