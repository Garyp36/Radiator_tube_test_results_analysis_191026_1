"""Local, unauthenticated saving of analysis outputs to ``saved_runs/``.

Storage on hosted platforms (e.g. Streamlit Community Cloud) can be ephemeral;
files may disappear on redeploy or restart. Nothing here is a substitute for a
database and no confidential data should be processed on a public deployment.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd

SAVED_RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_runs")


def ensure_dir(path: str = SAVED_RUNS_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitise_label(label: str, default: str = "run") -> str:
    """Keep letters, digits, dash and underscore only; collapse the rest to '_'."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (label or "").strip()).strip("_")
    return cleaned[:60] or default


def unique_prefix(label: str, directory: str = SAVED_RUNS_DIR) -> str:
    """``<label>_<YYYYmmdd_HHMMSS>`` with a numeric suffix if a collision occurs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{sanitise_label(label)}_{stamp}"
    candidate, counter = base, 1
    while any(name.startswith(candidate + "_") or name.startswith(candidate + ".")
              for name in os.listdir(directory)):
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


def save_run(label: str, summary: pd.DataFrame, processed: pd.DataFrame, settings: Dict[str, object],
             source_files: List[str], directory: str = SAVED_RUNS_DIR,
             extra_tables: Dict[str, pd.DataFrame] | None = None) -> List[Dict[str, object]]:
    """Write summary CSV, processed CSV and a settings JSON. Returns [{file, bytes}, ...]."""
    ensure_dir(directory)
    prefix = unique_prefix(label, directory)
    written: List[Dict[str, object]] = []

    def _write_csv(name: str, df: pd.DataFrame) -> None:
        path = os.path.join(directory, f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)
        written.append({"file": os.path.basename(path), "bytes": os.path.getsize(path)})

    _write_csv("summary", summary)
    _write_csv("processed", processed)
    for name, table in (extra_tables or {}).items():
        if table is not None and len(table):
            _write_csv(sanitise_label(name), table)

    meta = {
        "run_label": label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": list(source_files),
        "settings": settings,
        "files": [w["file"] for w in written],
    }
    meta_path = os.path.join(directory, f"{prefix}_settings.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    written.append({"file": os.path.basename(meta_path), "bytes": os.path.getsize(meta_path)})
    return written


def list_saved_runs(directory: str = SAVED_RUNS_DIR) -> pd.DataFrame:
    if not os.path.isdir(directory):
        return pd.DataFrame(columns=["file", "bytes", "modified"])
    rows = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and not name.startswith("."):
            rows.append({"file": name, "bytes": os.path.getsize(path),
                         "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")})
    return pd.DataFrame(rows, columns=["file", "bytes", "modified"])
