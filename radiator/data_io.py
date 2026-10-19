"""CSV loading, column alias mapping, validation and cleaning.

Nothing in this module fabricates measurements: invalid or missing readings are
either kept as NaN (optional columns) or the affected rows are discarded and
reported (essential columns).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Canonical column name -> accepted aliases (matched case-insensitively, with
# surrounding whitespace stripped and spaces/hyphens treated like underscores).
COLUMN_ALIASES: Dict[str, List[str]] = {
    "elapsed_s": ["elapsed_s", "elapsed", "seconds", "time_s", "elapsed_seconds", "t_s"],
    "datetime": ["datetime", "timestamp", "time", "date_time"],
    "T0_C": ["t0_c", "t0", "ambient", "ambient_c", "t_ambient", "ambient_temp"],
    "T1_C": ["t1_c", "t1", "heated_water", "heated_water_c", "vessel", "vessel_c"],
    "T2_C": ["t2_c", "t2", "inlet", "inlet_c", "t_inlet"],
    "T3_C": ["t3_c", "t3", "outlet", "outlet_c", "t_outlet"],
    "target_C": ["target_c", "target", "setpoint_c", "setpoint"],
    "heater_enabled": ["heater_enabled", "heater", "heater_on", "heater_state"],
    "ssr_cmd": ["ssr_cmd", "ssr", "ssr_command"],
    "fan_pwm_pct": ["fan_pwm_pct", "fan_pwm", "pwm", "pwm_pct"],
    "fan_rpm": ["fan_rpm", "rpm"],
    "air_speed_m_s": ["air_speed_m_s", "air_speed", "air_velocity", "air_velocity_m_s", "velocity_m_s"],
    "sensor_status": ["sensor_status", "sensors_status", "sensor"],
    "rtc_status": ["rtc_status", "rtc"],
    "heater_fault": ["heater_fault", "fault"],
}

ESSENTIAL_TEMPERATURES = ["T2_C", "T3_C"]
OPTIONAL_TEMPERATURES = ["T0_C", "T1_C"]
NUMERIC_COLUMNS = [
    "elapsed_s", "T0_C", "T1_C", "T2_C", "T3_C", "target_C",
    "fan_pwm_pct", "fan_rpm", "air_speed_m_s",
]

DUPLICATE_POLICY = (
    "Rows sharing an identical elapsed time are considered duplicate observations; "
    "the first occurrence (in file order) is retained and later duplicates are discarded."
)


@dataclass
class LoadResult:
    """Outcome of loading and cleaning one experiment CSV."""

    name: str
    raw: pd.DataFrame
    data: pd.DataFrame
    column_map: Dict[str, str] = field(default_factory=dict)
    detected: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    discarded: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_required and len(self.data) >= 2

    @property
    def has_ambient(self) -> bool:
        return "T0_C" in self.data.columns and self.data["T0_C"].notna().any()


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def map_columns(columns) -> Dict[str, str]:
    """Return {canonical_name: original_column_name} for recognised columns.

    Exact canonical matches win over aliases; the first alias match wins otherwise.
    """
    normalised = {_normalise(c): c for c in columns}
    mapping: Dict[str, str] = {}
    used = set()
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            original = normalised.get(_normalise(alias))
            if original is not None and original not in used:
                mapping[canonical] = original
                used.add(original)
                break
    return mapping


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce to float; non-numeric and infinite values become NaN."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.replace([np.inf, -np.inf], np.nan)


def elapsed_from_datetime(series: pd.Series) -> pd.Series:
    """Elapsed seconds relative to the earliest parseable timestamp (NaN if unparseable)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    start = parsed.min()
    return (parsed - start).dt.total_seconds().astype(float)


def load_experiment_csv(source: Union[str, IO], name: Optional[str] = None) -> LoadResult:
    """Read a CSV file (path or file-like) and return a cleaned, validated LoadResult."""
    if name is None:
        name = getattr(source, "name", None) or str(source)
    try:
        raw = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - depends on malformed input
        empty = pd.DataFrame()
        return LoadResult(name=name, raw=empty, data=empty,
                          missing_required=["elapsed_s", "T2_C", "T3_C"],
                          warnings=[f"Could not parse CSV: {exc}"])
    return clean_dataframe(raw, name=name)


def clean_dataframe(raw: pd.DataFrame, name: str = "experiment") -> LoadResult:
    """Map aliases, coerce numerics, derive elapsed time, sort, de-duplicate and validate."""
    raw = raw.copy()
    # Drop fully empty columns/rows that spreadsheets often append.
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mapping = map_columns(raw.columns)
    warnings: List[str] = []
    discarded: Dict[str, int] = {}

    df = pd.DataFrame(index=raw.index)
    for canonical, original in mapping.items():
        df[canonical] = raw[original]
    # Preserve unrecognised metadata columns untouched.
    for col in raw.columns:
        if col not in mapping.values():
            df[f"extra:{col}"] = raw[col]

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # Elapsed time: prefer explicit elapsed column; otherwise derive from timestamps.
    if "elapsed_s" in df.columns and df["elapsed_s"].notna().any():
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "datetime" in df.columns:
        df["elapsed_s"] = elapsed_from_datetime(df["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        warnings.append("No elapsed-time column found; elapsed_s was derived from the timestamp column.")
    else:
        df["elapsed_s"] = np.nan

    missing_required = []
    if df["elapsed_s"].notna().sum() == 0:
        missing_required.append("elapsed_s (or parseable datetime)")
    for col in ESSENTIAL_TEMPERATURES:
        if col not in df.columns:
            missing_required.append(col)
    missing_optional = [c for c in OPTIONAL_TEMPERATURES + ["air_speed_m_s", "fan_pwm_pct", "fan_rpm"]
                        if c not in df.columns]

    detected = [c for c in COLUMN_ALIASES if c in df.columns]

    if missing_required:
        return LoadResult(name=name, raw=raw, data=df.iloc[0:0], column_map=mapping, detected=detected,
                          missing_required=missing_required, missing_optional=missing_optional,
                          warnings=warnings, discarded=discarded)

    n0 = len(df)
    df = df[df["elapsed_s"].notna()]
    discarded["invalid_elapsed_time"] = n0 - len(df)

    n1 = len(df)
    df = df[df["T2_C"].notna() & df["T3_C"].notna()]
    discarded["invalid_T2_or_T3"] = n1 - len(df)

    df = df.sort_values("elapsed_s", kind="mergesort")
    n2 = len(df)
    df = df[~df["elapsed_s"].duplicated(keep="first")]
    discarded["duplicate_elapsed_time"] = n2 - len(df)
    if discarded["duplicate_elapsed_time"]:
        warnings.append(f"{discarded['duplicate_elapsed_time']} duplicate elapsed-time row(s) discarded. "
                        + DUPLICATE_POLICY)
    for key, count in discarded.items():
        if count and key != "duplicate_elapsed_time":
            warnings.append(f"{count} row(s) discarded because of {key.replace('_', ' ')}.")

    df = df.reset_index(drop=True)
    if len(df) < 2:
        warnings.append("Fewer than two valid readings remain; cooling analysis is not possible.")
    if "T0_C" not in df.columns:
        warnings.append("No ambient (T0) column detected; the constant ambient fallback will be used.")
    elif df["T0_C"].isna().any():
        warnings.append(f"{int(df['T0_C'].isna().sum())} reading(s) have no T0 value; the ambient fallback "
                        "will be substituted for those rows.")

    return LoadResult(name=name, raw=raw, data=df, column_map=mapping, detected=detected,
                      missing_required=missing_required, missing_optional=missing_optional,
                      warnings=warnings, discarded=discarded)


def metadata_from_data(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Representative values of optional operating-condition columns (median; NaN -> None)."""
    out: Dict[str, Optional[float]] = {}
    for col in ("air_speed_m_s", "fan_pwm_pct", "fan_rpm"):
        if col in df.columns and df[col].notna().any():
            out[col] = float(df[col].median())
        else:
            out[col] = None
    return out
