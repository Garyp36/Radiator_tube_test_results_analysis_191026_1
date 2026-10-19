import json
import os

import pandas as pd

from radiator.calculations import ExperimentSettings, build_processed_table, summarise_experiment
from radiator.comparison import build_comparison_table, common_time_comparison, threshold_comparison
from radiator.fitting import fit_exponential, fit_linear
from radiator.storage import sanitise_label, save_run


def _exp(name, times, t2, t3, ambient=30.0):
    df = pd.DataFrame({"elapsed_s": times, "T0_C": [ambient] * len(times), "T2_C": t2, "T3_C": t3})
    return build_processed_table(df, ExperimentSettings(), name)


def test_different_duration_experiments_compare_without_extrapolation():
    short = _exp("short", [0, 600, 1200], [64, 60, 56], [54, 50, 46])
    long = _exp("long", [0, 600, 1200, 1800, 2400], [64, 62, 60, 58, 56], [54, 52, 50, 48, 46])
    proc = {"short": short, "long": long}

    ct = common_time_comparison(proc, 30.0).set_index("experiment")  # 1800 s
    assert pd.isna(ct.loc["short", "Tmean_C"]) and not ct.loc["short", "within_range"]
    assert ct.loc["long", "Tmean_C"] == 53.0 and ct.loc["long", "within_range"]

    th = threshold_comparison(proc, "Tmean_C", 52.0).set_index("experiment")
    assert th.loc["short", "status"] == "crossed" and th.loc["short", "time_to_threshold_min"] == 17.5  # 600 s + 0.75*600 s
    assert th.loc["long", "status"] == "crossed" and th.loc["long", "time_to_threshold_min"] == 35.0  # 1800 s + 0.5*600 s
    th2 = threshold_comparison(proc, "Tmean_C", 40.0).set_index("experiment")
    assert (th2["status"] == "not_reached").all() and (th2["result"] == "Not reached").all()
    th3 = threshold_comparison(proc, "theta_C", 40.0).set_index("experiment")
    assert (th3["status"] == "already_below_at_start").all()


def test_comparison_table_contains_fit_metrics():
    p = _exp("a", [0, 600, 1200, 1800, 2400], [64, 62, 60, 58, 56], [54, 52, 50, 48, 46])
    s = summarise_experiment(p, ExperimentSettings(air_velocity_m_s=1.1), "a")
    fits = {"a": {"linear": fit_linear(p["elapsed_min"], p["Tmean_C"]),
                  "exponential": fit_exponential(p["elapsed_min"], p["theta_C"])}}
    table = build_comparison_table([s], fits)
    assert table.loc[0, "experiment"] == "a"
    assert table.loc[0, "linear_r2"] > 0.999
    assert table.loc[0, "air_velocity_m_s"] == 1.1
    assert "apparent_heat_flux_W_m2" in table.columns


def test_sanitise_label():
    assert sanitise_label("My run / test #1!") == "My_run_test_1"
    assert sanitise_label("") == "run"
    assert sanitise_label("../../etc") == "etc"


def test_save_run_writes_unique_files(tmp_path):
    summary = pd.DataFrame({"a": [1]})
    processed = pd.DataFrame({"b": [1, 2]})
    first = save_run("demo", summary, processed, {"x": 1}, ["f.csv"], directory=str(tmp_path))
    second = save_run("demo", summary, processed, {"x": 1}, ["f.csv"], directory=str(tmp_path))
    names = {w["file"] for w in first} | {w["file"] for w in second}
    assert len(names) == 6  # no overwrite: 3 files per save, all distinct
    for w in first + second:
        assert os.path.getsize(tmp_path / w["file"]) == w["bytes"] > 0
    meta = [w for w in first if w["file"].endswith(".json")][0]
    with open(tmp_path / meta["file"]) as fh:
        data = json.load(fh)
    assert data["source_files"] == ["f.csv"] and data["settings"] == {"x": 1}
