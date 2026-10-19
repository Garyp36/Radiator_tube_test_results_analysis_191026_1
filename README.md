# Radiator Tube Cooling Analysis (Streamlit)

*Heat Transfer | Heat Flux | Cooling Rates | Curve Fitting | Experiment Comparison*

A Streamlit dashboard for analysing stationary-water cooling tests on a short aluminium
radiator tube with a fan supplying air across it. It is written for PhD research on
heat-transfer performance and tube-design optimisation and runs locally or on
Streamlit Community Cloud without any API keys, accounts or databases.

> **Scope statement.** T2 and T3 are the water temperatures at the two *ends* of the tube.
> Their arithmetic mean `Tmean = (T2 + T3)/2` is an **approximation** of the representative
> water temperature, not a measured bulk mean. The app estimates the **sensible-energy change
> of the water** and derives *apparent* heat-loss rates and heat fluxes from it. It does **not**
> measure a steady-state radiator heat-transfer coefficient or a direct tube-wall heat flux.

---

## 1. Project layout

```
radiator_streamlit_app/
├── app.py                     # Streamlit UI (single analysis + comparison)
├── radiator/
│   ├── __init__.py
│   ├── data_io.py             # CSV loading, column aliases, validation, cleaning
│   ├── calculations.py        # mass, geometry, energy, power, flux, cooling rates,
│   │                          # processed table, summary, threshold crossing, interpolation
│   ├── fitting.py             # linear and exponential (SciPy curve_fit) models
│   ├── comparison.py          # comparison table and common-criterion comparisons
│   ├── plots.py               # Plotly figures
│   └── storage.py             # saved_runs/ local storage
├── tests/                     # pytest suite (37 tests)
├── sample_experiment.csv      # bundled sample data (NOT a real experimental result)
├── sample_experiment_b.csv    # second sample for the comparison mode
├── saved_runs/                # created automatically; local saves land here
├── requirements.txt
└── README.md
```

## 2. Installation

Recommended: **Python 3.10 – 3.12** (tested with 3.10).

**Windows (PowerShell / cmd)**
```bat
cd radiator_streamlit_app
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd radiator_streamlit_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Launch locally

```bash
streamlit run app.py
```
(Windows: same command inside the activated virtual environment.) The browser opens at
<http://localhost:8501>.

Run the automated tests with
```bash
python -m pytest -q
```

## 4. Using the app

1. In the sidebar choose **Single Experiment Analysis** or **Experiment Comparison**.
2. Upload one or more CSV files (or tick *Use bundled sample data*).
3. Check the **Column detection and data quality** panel: it lists required/optional columns
   found or missing, how many rows were discarded and why, and whether the ambient fallback
   will be used.
4. Adjust **Common experiment settings** (water, ambient fallback, geometry, area method, air
   velocity). For each experiment an expander in the main panel lets you **override** these
   values individually – necessary when experiments use different tube designs or water volumes.
   Nothing is silently shared: the active values are printed in the summary table.
5. Optionally enable a **report interval** (nearest actual observations every *N* seconds) and
   choose whether metrics/fits use *all valid readings* or the *interval-reduced* readings, and/or
   restrict the **analysis time window**. Original readings are always retained and downloadable.
6. Read the metric cards, summary table, plots, fit statistics and the processed table; download
   CSVs; optionally save to `saved_runs/`.

### Accepted CSV columns

Preferred header names:

`elapsed_s, datetime, T0_C, T1_C, T2_C, T3_C, target_C, heater_enabled, ssr_cmd, fan_pwm_pct, fan_rpm, air_speed_m_s, sensor_status, rtc_status, heater_fault`

Essential: `elapsed_s` (or a parseable `datetime`/`timestamp` from which elapsed time is derived),
`T2_C`, `T3_C`. Everything else is optional; analyses depending on an absent optional column are
labelled/disabled rather than faked.

Aliases (case-insensitive, spaces/hyphens treated as underscores):

| Canonical | Aliases |
|---|---|
| `elapsed_s` | `elapsed`, `seconds`, `time_s`, `elapsed_seconds` |
| `datetime` | `timestamp`, `time`, `date_time` |
| `T0_C` | `T0`, `ambient`, `ambient_C`, `t_ambient` |
| `T1_C` | `T1`, `heated_water`, `heated_water_C`, `vessel` |
| `T2_C` | `T2`, `inlet`, `inlet_C`, `t_inlet` |
| `T3_C` | `T3`, `outlet`, `outlet_C`, `t_outlet` |
| `air_speed_m_s` | `air_speed`, `air_velocity`, `velocity_m_s` |
| `fan_pwm_pct` | `fan_pwm`, `pwm` ; `fan_rpm` | `rpm` |

Cleaning policy: numeric columns are coerced (text, blanks, ±inf → missing); rows without a valid
elapsed time or without both T2 and T3 are discarded and counted; rows are sorted by elapsed time;
**duplicate elapsed times keep the first occurrence** (in file order) and later duplicates are
discarded. Missing T0 values are never invented – the configurable constant ambient fallback is
substituted *per reading* and flagged in the table (`ambient_source`), the summary and a warning.

## 5. Calculations, units and assumptions

| Quantity | Formula | Unit |
|---|---|---|
| Mean water temperature (endpoint mean) | `Tmean = (T2 + T3) / 2` | °C |
| Endpoint gap | `abs(T2 − T3)`; mean and maximum over the window | °C |
| Water mass (if not measured) | `m = V[mL] × 1e-6 × ρ[kg/m³]` – V is the *complete* inventory (tube + silicone) | kg |
| Interval cooling rate (T2, T3, Tmean) | `−ΔT / Δt × 3600` – signed, negative if the water warmed | °C/h |
| Window average cooling rate | `−(T_end − T_start) / duration × 3600` | °C/h |
| Cumulative sensible energy removed | `m × cp × (Tmean,initial − Tmean,current)` | J |
| Interval energy removed | `m × cp × (Tmean,previous − Tmean,current)` | J |
| Interval net heat-loss rate | `interval energy / interval duration` | W |
| Average net heat-loss rate | `total energy removed / total duration` | W |
| Exposed area (rectangular tube) | `2 × (w + h) × L` with mm → m; or manual value | m² |
| Apparent average heat flux | `average heat-loss rate / exposed area` | W/m² |
| Interval apparent heat flux | `interval heat-loss rate / exposed area` | W/m² |
| Cumulative-average heat-flux equivalent | `cumulative energy / elapsed since start / area` | W/m² |
| Temperature excess | `θ = Tmean − T0(used)` ; normalised `θ / θ₀` (blank if θ₀ = 0) | °C, – |

**Energy vs power vs heat flux.** Energy (J) is an amount of heat. Power (W = J/s) is the rate
at which that heat is lost – interval values are averages over each interval, the overall value
is total energy over total duration. Heat flux (W/m²) is power per unit exposed area.
Interval-average rates are *not* instantaneous measured rates.

Temperature differences in °C equal those in K, so `cp` in J/(kg·K) is applied directly.

### Curve fits
* **Linear:** `Tmean = intercept + slope × t` (t in minutes). Reported: equation, slope (°C/min),
  equivalent cooling rate (°C/h = −slope × 60), R², RMSE, predicted values and residuals.
* **Exponential:** `θ(t) = A × exp(−k × t)` fitted to the ambient-corrected excess with SciPy
  `curve_fit` (A > 0, −1 ≤ k ≤ 10 min⁻¹). Reported: A, k (min⁻¹), τ = 1/k (min, only when k > 0),
  R², RMSE, fitted curve and residuals. Requires ≥ 4 points with positive θ; failures are
  reported, never fabricated.
* R²/RMSE describe how closely a curve follows the data; a good fit does not prove the model is
  physically correct (tube-wall heat storage, sensor lag and vessel exchange are not modelled).
* The fit section states whether *all valid readings* or *interval-reduced readings* were used.

### Experiment comparison
Each experiment carries its own settings. The comparison table lists design, air velocity, fan
PWM/RPM, duration, water mass, initial/final Tmean, temperature drop, T2/T3/Tmean cooling rates,
energy removed, average heat-loss rate, exposed area, apparent heat flux, mean/max |T2−T3| and
both fits' R²/RMSE. Three common criteria:

1. **Time to a common Tmean threshold** and 2. **time to a common θ threshold** – first downward
   crossing within recorded data, linearly interpolated between the bracketing readings, never
   extrapolated. Already below at start → `0 min (already below threshold at start)`; never
   reached → `Not reached`.
3. **State at a common elapsed time** – interpolated only inside each experiment's measured range;
   blank otherwise. Each experiment's measured range is displayed.

Overlay plots: T2/T3/Tmean, θ, cumulative energy, with experiment selection.

### Report interval and windows
With the report interval enabled, the app selects the **nearest actual observation** to each
target time (start, start + N, …), always keeps the first and last observation, removes
duplicates and never creates synthetic rows. Full-resolution processed data and the reduced
table are both downloadable. A time window restricts the readings used for all window metrics
while the original readings remain available.

## 6. Local storage
The **Save** button writes `<label>_<timestamp>_summary.csv`, `…_processed.csv`, optional
`…_fits.csv`/`…_residuals.csv`/`…_comparison.csv` and `…_settings.json` (settings, timestamps,
source file names) into `saved_runs/`, which is created automatically. Labels are sanitised and
names are made unique, so nothing is overwritten. Saved file names and sizes are displayed.

**Hosting caveat:** on Streamlit Community Cloud the file system is ephemeral – saved files can
vanish on restart or redeploy. Download the CSVs if you need them.
**Privacy:** there is no authentication. Do not deploy publicly with confidential data.

## 7. Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository (keep `app.py` at the root, or note its path).
2. Sign in at <https://share.streamlit.io>, click **New app**, select the repo/branch and set
   *Main file path* to `app.py`.
3. Python version: choose 3.10–3.12 in *Advanced settings*. Dependencies are read from
   `requirements.txt` automatically.
4. Deploy. No secrets or API keys are required.

## 8. Known limitations
* Energy balance uses the water's sensible heat only; heat stored in the aluminium tube and
  silicone passages, heat exchange with the heated vessel or surroundings, radiation and sensor
  response time are not separated out. Results are therefore *apparent/estimated* values.
* `Tmean` is the mean of two endpoint sensors, not a measured bulk temperature; large |T2−T3|
  gaps indicate strong axial gradients where this approximation is poor.
* The geometric area `2(w+h)L` assumes all four long faces are exposed and ignores end faces,
  fins, fittings, insulation or partial coverage; use the manual-area option when appropriate.
* Finite-difference interval rates amplify sensor noise and resolution steps (e.g. 1 °C data).
* Threshold crossings assume the signal is linear between readings.
* Curve-fit parameter uncertainty is not reported.
* Streamlit `number_input` cannot be empty, so `0` means "not provided" for measured mass,
  manual area and air velocity.

## 9. Testing performed
`python -m pytest -q` – 37 tests covering energy against a hand-calculated example,
volume→mass, rectangular area, interval/total heat-loss and heat-flux, cooling-rate signs,
missing columns, invalid readings, duplicate timestamps, zero-duration intervals, linear and
exponential fits on synthetic data, threshold crossings (crossed, already met, not reached,
NaN-containing), common-time interpolation without extrapolation, sample CSV loading, local
storage uniqueness. `python -m py_compile` on all sources and a headless `streamlit run`
smoke test were also executed.
