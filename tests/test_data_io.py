import io
import os

import numpy as np
import pandas as pd

from radiator.data_io import clean_dataframe, load_experiment_csv, map_columns, metadata_from_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sample_csv_loads():
    res = load_experiment_csv(os.path.join(ROOT, "sample_experiment.csv"))
    assert res.ok and not res.missing_required
    assert len(res.data) == 10
    assert res.data["T2_C"].iloc[0] == 64 and res.data["T3_C"].iloc[-1] == 30
    assert res.data["elapsed_s"].iloc[-1] == 5400
    assert metadata_from_data(res.data)["air_speed_m_s"] == 1.1


def test_second_sample_csv_loads():
    res = load_experiment_csv(os.path.join(ROOT, "sample_experiment_b.csv"))
    assert res.ok and len(res.data) == 8


def test_column_aliases():
    mapping = map_columns(["Elapsed", "inlet", "Outlet_C", "ambient", "heated_water", "Air Speed"])
    assert mapping["elapsed_s"] == "Elapsed"
    assert mapping["T2_C"] == "inlet"
    assert mapping["T3_C"] == "Outlet_C"
    assert mapping["T0_C"] == "ambient"
    assert mapping["T1_C"] == "heated_water"
    assert mapping["air_speed_m_s"] == "Air Speed"


def test_missing_required_columns_reported():
    res = clean_dataframe(pd.DataFrame({"elapsed_s": [0, 1], "T2_C": [1, 2]}))
    assert not res.ok and "T3_C" in res.missing_required
    res = clean_dataframe(pd.DataFrame({"T2_C": [1, 2], "T3_C": [1, 2]}))
    assert any("elapsed" in m for m in res.missing_required)


def test_elapsed_derived_from_timestamp():
    df = pd.DataFrame({"timestamp": ["2025-01-01 10:00:00", "2025-01-01 10:10:00"], "T2": [60, 58], "T3": [50, 48]})
    res = clean_dataframe(df)
    assert res.ok
    assert res.data["elapsed_s"].tolist() == [0.0, 600.0]
    assert any("derived" in w for w in res.warnings)


def test_invalid_readings_discarded_and_reported():
    csv = "elapsed_s,T0_C,T2_C,T3_C\n0,30,64,54\n600,30,abc,51\n1200,,60,inf\n1800,30,58,42\n,30,57,38\n2400,30,57,38\n"
    res = load_experiment_csv(io.StringIO(csv), name="bad.csv")
    assert res.ok
    assert res.discarded["invalid_T2_or_T3"] == 2
    assert res.discarded["invalid_elapsed_time"] == 1
    assert res.data["elapsed_s"].tolist() == [0, 1800, 2400]
    # Optional columns keep NaN rather than fabricated values
    assert not res.data["T0_C"].isna().any()  # the NaN T0 row was discarded for other reasons here
    assert len(res.warnings) >= 2


def test_optional_nan_kept_and_flagged():
    df = pd.DataFrame({"elapsed_s": [0, 600], "T0_C": [30, np.nan], "T2_C": [60, 58], "T3_C": [50, 48]})
    res = clean_dataframe(df)
    assert res.ok and res.data["T0_C"].isna().sum() == 1
    assert any("fallback" in w for w in res.warnings)


def test_duplicate_timestamps_first_kept_and_sorted():
    df = pd.DataFrame({"elapsed_s": [600, 0, 600, 1200], "T2_C": [62, 64, 99, 60], "T3_C": [51, 54, 99, 48]})
    res = clean_dataframe(df)
    assert res.data["elapsed_s"].tolist() == [0, 600, 1200]
    assert res.data["T2_C"].tolist() == [64, 62, 60]
    assert res.discarded["duplicate_elapsed_time"] == 1
    assert any("duplicate" in w for w in res.warnings)


def test_metadata_columns_preserved():
    df = pd.DataFrame({"elapsed_s": [0, 600], "T2_C": [60, 58], "T3_C": [50, 48], "sensor_status": ["OK", "OK"],
                       "my_note": ["a", "b"]})
    res = clean_dataframe(df)
    assert "sensor_status" in res.data.columns and "extra:my_note" in res.data.columns
