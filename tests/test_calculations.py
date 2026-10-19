import math

import numpy as np
import pandas as pd
import pytest

from radiator.calculations import (ExperimentSettings, apply_time_window, build_processed_table, cooling_rate_C_per_h,
                                   first_downward_crossing, heat_flux_W_m2, heat_loss_rate_W,
                                   rectangular_tube_area_m2, reduce_to_report_interval, resolve_exposed_area,
                                   sensible_energy_J, summarise_experiment, value_at_time, water_mass_kg)


def test_volume_to_mass_conversion():
    # 20 mL * 1e-6 m3/mL * 998 kg/m3 = 0.01996 kg
    assert water_mass_kg(20.0, 998.0) == pytest.approx(0.01996)
    assert water_mass_kg(1000.0, 1000.0) == pytest.approx(1.0)


def test_measured_mass_overrides_volume():
    assert water_mass_kg(20.0, 998.0, measured_mass_kg=0.025) == 0.025
    assert water_mass_kg(20.0, 998.0, measured_mass_kg=0.0) == pytest.approx(0.01996)
    assert water_mass_kg(20.0, 998.0, measured_mass_kg=None) == pytest.approx(0.01996)


def test_invalid_volume_raises():
    with pytest.raises(ValueError):
        water_mass_kg(0.0, 998.0)


def test_rectangular_tube_area():
    # 2 * (0.016 + 0.003) * 0.100 = 0.0038 m2
    assert rectangular_tube_area_m2(100.0, 16.0, 3.0) == pytest.approx(0.0038)
    with pytest.raises(ValueError):
        rectangular_tube_area_m2(100.0, 0.0, 3.0)


def test_manual_area_selection():
    area, desc = resolve_exposed_area("manual", 100, 16, 3, 0.005)
    assert area == 0.005 and "Manual" in desc
    area, desc = resolve_exposed_area("perimeter", 100, 16, 3, None)
    assert area == pytest.approx(0.0038)
    with pytest.raises(ValueError):
        resolve_exposed_area("manual", 100, 16, 3, None)


def test_sensible_energy_independent_example():
    # Independent hand calculation: 0.01996 kg * 4180 J/kgK * (59 - 41.5) K = 1460.074 J
    assert sensible_energy_J(0.01996, 4180.0, 59.0, 41.5) == pytest.approx(1460.074, rel=1e-9)
    # Warming water -> negative energy removed
    assert sensible_energy_J(0.02, 4180.0, 50.0, 52.0) == pytest.approx(-167.2)


def test_heat_loss_rate_and_flux():
    assert heat_loss_rate_W(1460.074, 5400.0) == pytest.approx(0.270384, rel=1e-5)
    assert math.isnan(heat_loss_rate_W(100.0, 0.0))
    assert math.isnan(heat_loss_rate_W(100.0, float("nan")))
    assert heat_flux_W_m2(0.2703840740740741, 0.0038) == pytest.approx(71.1537, rel=1e-5)
    assert math.isnan(heat_flux_W_m2(1.0, 0.0))


def test_cooling_rate_sign_and_units():
    # 10 degC drop over 1 hour = +10 degC/h
    assert cooling_rate_C_per_h(60.0, 50.0, 3600.0) == pytest.approx(10.0)
    # warming is negative, not clipped
    assert cooling_rate_C_per_h(50.0, 52.0, 1800.0) == pytest.approx(-4.0)
    assert math.isnan(cooling_rate_C_per_h(50.0, 40.0, 0.0))


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "elapsed_s": [0, 600, 1200, 1800],
        "T0_C": [30.0, 30.0, np.nan, 30.0],
        "T2_C": [64.0, 62.0, 60.0, 61.0],
        "T3_C": [54.0, 51.0, 48.0, 49.0],
    })


def test_processed_table_values(sample_df):
    settings = ExperimentSettings(ambient_fallback_C=29.0)
    warnings = []
    p = build_processed_table(sample_df, settings, "x", warnings)
    mass = 0.01996
    assert p["Tmean_C"].tolist() == [59.0, 56.5, 54.0, 55.0]
    assert p["T_gap_abs_C"].tolist() == [10.0, 11.0, 12.0, 12.0]
    # First row: interval quantities blank, cumulative zero
    assert np.isnan(p.loc[0, "interval_energy_J"]) and np.isnan(p.loc[0, "rate_Tmean_C_per_h"])
    assert p.loc[0, "cumulative_energy_J"] == 0.0
    assert p.loc[1, "interval_energy_J"] == pytest.approx(mass * 4180 * 2.5)
    assert p.loc[1, "interval_heat_loss_W"] == pytest.approx(mass * 4180 * 2.5 / 600)
    assert p.loc[1, "interval_heat_flux_W_m2"] == pytest.approx(mass * 4180 * 2.5 / 600 / 0.0038)
    assert p.loc[1, "rate_T2_C_per_h"] == pytest.approx(12.0)
    assert p.loc[1, "rate_T3_C_per_h"] == pytest.approx(18.0)
    # Interval with warming -> negative interval energy and negative rate
    assert p.loc[3, "interval_energy_J"] < 0 and p.loc[3, "rate_Tmean_C_per_h"] == pytest.approx(-6.0)
    assert p.loc[3, "cumulative_energy_J"] == pytest.approx(mass * 4180 * 4.0)
    # Ambient fallback used only where T0 missing, and flagged
    assert p["ambient_source"].tolist() == ["measured T0", "measured T0", "fallback", "measured T0"]
    assert p.loc[2, "ambient_used_C"] == 29.0
    assert p.loc[2, "theta_C"] == pytest.approx(25.0)
    assert any("fallback" in w for w in warnings)
    assert p["theta_norm"].iloc[0] == pytest.approx(1.0)
    assert p["theta_norm"].iloc[1] == pytest.approx(26.5 / 29.0)


def test_summary_total_and_interval_average_consistency(sample_df):
    settings = ExperimentSettings()
    p = build_processed_table(sample_df, settings, "x")
    s = summarise_experiment(p, settings, "x")
    assert s["duration_s"] == 1800
    assert s["energy_removed_J"] == pytest.approx(p["interval_energy_J"].iloc[1:].sum())
    assert s["energy_removed_J"] == pytest.approx(0.01996 * 4180 * 4.0)
    assert s["avg_heat_loss_W"] == pytest.approx(s["energy_removed_J"] / 1800)
    assert s["apparent_heat_flux_W_m2"] == pytest.approx(s["avg_heat_loss_W"] / 0.0038)
    assert s["rate_Tmean_C_per_h"] == pytest.approx((59.0 - 55.0) / 1800 * 3600)
    assert s["gap_mean_C"] == pytest.approx(11.25) and s["gap_max_C"] == 12.0
    assert s["ambient_source"] == "mixed (measured + fallback)"


def test_zero_duration_interval_is_blank_not_infinite():
    df = pd.DataFrame({"elapsed_s": [0, 0, 600], "T2_C": [60, 59, 58], "T3_C": [50, 49, 48]})
    warnings = []
    p = build_processed_table(df, ExperimentSettings(), "x", warnings)
    assert np.isnan(p.loc[1, "interval_dt_s"]) and np.isnan(p.loc[1, "rate_T2_C_per_h"])
    assert np.isnan(p.loc[1, "interval_heat_loss_W"])
    assert np.isfinite(p.loc[2, "interval_heat_loss_W"])
    assert any("zero duration" in w for w in warnings)


def test_zero_initial_theta_gives_nan_normalised():
    df = pd.DataFrame({"elapsed_s": [0, 600], "T0_C": [30.0, 30.0], "T2_C": [30, 29], "T3_C": [30, 29]})
    warnings = []
    p = build_processed_table(df, ExperimentSettings(), "x", warnings)
    assert p["theta_norm"].isna().all()
    assert any("θ₀" in w for w in warnings)


def test_summary_with_single_reading_is_nan_not_crash():
    df = pd.DataFrame({"elapsed_s": [0], "T2_C": [60.0], "T3_C": [50.0]})
    p = build_processed_table(df, ExperimentSettings(), "x")
    s = summarise_experiment(p, ExperimentSettings(), "x")
    assert math.isnan(s["energy_removed_J"]) and math.isnan(s["duration_s"])


def test_time_window():
    df = pd.DataFrame({"elapsed_s": [0, 600, 1200, 1800], "T2_C": [1, 2, 3, 4], "T3_C": [1, 2, 3, 4]})
    assert apply_time_window(df, 600, 1200)["elapsed_s"].tolist() == [600, 1200]
    assert apply_time_window(df, None, 600)["elapsed_s"].tolist() == [0, 600]
    assert len(apply_time_window(df, 5000, None)) == 0


def test_report_interval_reduction_selects_real_rows():
    times = [0, 290, 610, 905, 1190, 1500, 1790, 2100, 2350]
    df = pd.DataFrame({"elapsed_s": times, "T2_C": range(9), "T3_C": range(9)})
    reduced = reduce_to_report_interval(df, 600)
    assert reduced["elapsed_s"].tolist() == [0, 610, 1190, 1790, 2350]
    assert set(reduced["elapsed_s"]).issubset(set(times))  # no synthetic rows
    assert not reduced["elapsed_s"].duplicated().any()
    # First and last retained even when interval larger than span
    assert reduce_to_report_interval(df, 10000)["elapsed_s"].tolist() == [0, 2350]
    assert reduce_to_report_interval(df, 0)["elapsed_s"].tolist() == times


def test_threshold_crossing_interpolated():
    t = [0, 600, 1200, 1800]
    v = [60.0, 56.0, 52.0, 50.0]
    res = first_downward_crossing(t, v, 54.0)
    assert res.status == "crossed" and res.time_s == pytest.approx(900.0)
    res = first_downward_crossing(t, v, 50.0)  # hits exactly the last reading
    assert res.status == "crossed" and res.time_s == pytest.approx(1800.0)


def test_threshold_already_met_and_not_reached():
    t = [0, 600, 1200]
    v = [60.0, 56.0, 52.0]
    res = first_downward_crossing(t, v, 65.0)
    assert res.status == "already_below_at_start" and res.time_s == 0.0
    res = first_downward_crossing(t, v, 40.0)
    assert res.status == "not_reached" and res.time_s is None and res.label() == "Not reached"
    assert first_downward_crossing([], [], 40.0).status == "insufficient_data"


def test_threshold_ignores_invalid_readings_and_offsets_start():
    t = [100, 700, 1300]
    v = [60.0, np.nan, 50.0]
    res = first_downward_crossing(t, v, 55.0)
    assert res.status == "crossed" and res.time_s == pytest.approx(600.0)  # relative to first valid reading


def test_common_time_value_no_extrapolation():
    t = [0, 600, 1200]
    v = [60.0, 56.0, 52.0]
    assert value_at_time(t, v, 300) == pytest.approx(58.0)
    assert value_at_time(t, v, 1200) == pytest.approx(52.0)
    assert math.isnan(value_at_time(t, v, 1800))  # beyond range -> missing, not extrapolated
    assert math.isnan(value_at_time(t, v, -10))
