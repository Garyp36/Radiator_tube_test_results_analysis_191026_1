import math

import numpy as np

from radiator.fitting import fit_exponential, fit_linear, predict_exponential, predict_linear


def test_linear_fit_recovers_known_parameters():
    t = np.arange(0, 60, 5.0)
    y = 58.0 - 0.2 * t
    fit = fit_linear(t, y)
    assert fit["ok"]
    np.testing.assert_allclose(fit["intercept_C"], 58.0, rtol=1e-9)
    np.testing.assert_allclose(fit["slope_C_per_min"], -0.2, rtol=1e-9)
    np.testing.assert_allclose(fit["cooling_rate_C_per_h"], 12.0, rtol=1e-9)
    assert fit["r2"] > 0.999999 and fit["rmse"] < 1e-9
    np.testing.assert_allclose(predict_linear(fit, [10.0]), [56.0])


def test_exponential_fit_recovers_known_parameters():
    t = np.linspace(0, 90, 19)
    theta = 29.0 * np.exp(-0.012 * t)
    fit = fit_exponential(t, theta)
    assert fit["ok"]
    np.testing.assert_allclose(fit["A_C"], 29.0, rtol=1e-4)
    np.testing.assert_allclose(fit["k_per_min"], 0.012, rtol=1e-4)
    np.testing.assert_allclose(fit["tau_min"], 1 / 0.012, rtol=1e-4)
    assert fit["r2"] > 0.99999
    np.testing.assert_allclose(predict_exponential(fit, [0.0]), [29.0], rtol=1e-4)


def test_exponential_fit_with_noise_reports_metrics():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 90, 30)
    theta = 25.0 * np.exp(-0.02 * t) + rng.normal(0, 0.2, t.size)
    fit = fit_exponential(t, theta)
    assert fit["ok"] and 0.9 < fit["r2"] <= 1.0 and 0.05 < fit["rmse"] < 0.5
    assert len(fit["fitted"]) == 30


def test_fits_handle_insufficient_data():
    assert not fit_linear([0, 1], [1, 2])["ok"]
    assert not fit_exponential([0, 1, 2], [3, 2, 1])["ok"]
    bad = fit_exponential([0, 1, 2, 3], [-1, -2, -3, -4])
    assert not bad["ok"] and math.isnan(bad["r2"])
    assert not fit_linear([0, 0, 0], [1, 2, 3])["ok"]


def test_fits_ignore_nan():
    t = [0, 10, 20, 30, 40]
    y = [50, np.nan, 46, 44, 42]
    fit = fit_linear(t, y)
    assert fit["ok"] and fit["n_points"] == 4
