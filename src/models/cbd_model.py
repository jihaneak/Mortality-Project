# src/models/cbd_model.py
import numpy as np
import pandas as pd
import pmdarima as pm
from src.models.life_expectancy import compute_life_table

def fit_cbd(df, age_col='Age', year_col='Year', mx_col='mx', age_min=50, age_max=90):
    """
    Cairns-Blake-Dowd model: logit(qx) = k1(t) + k2(t)*(x - xbar)
    Fitted on ages 40-90 where the linear logit assumption holds.
    Returns k1, k2 as Series indexed by year, xbar as scalar.
    """
    # Convert mx to qx
    df_fit = df[(df[age_col] >= age_min) & (df[age_col] <= age_max)].copy()
    df_fit['qx'] = df_fit[mx_col] / (1 + 0.5 * df_fit[mx_col])
    df_fit['qx'] = df_fit['qx'].clip(1e-6, 1 - 1e-6)
    df_fit['logit_qx'] = np.log(df_fit['qx'] / (1 - df_fit['qx']))

    df_pivot = df_fit.pivot(index=age_col, columns=year_col, values='logit_qx')
    ages  = df_pivot.index.values.astype(float)
    years = df_pivot.columns
    xbar  = ages.mean()

    k1_vals, k2_vals = [], []

    for year in years:
        y = df_pivot[year].values          # logit(qx) for each age
        X = np.column_stack([              # design matrix
            np.ones(len(ages)),
            ages - xbar
        ])
        # OLS: [k1, k2] = (X'X)^{-1} X'y
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        k1_vals.append(coeffs[0])
        k2_vals.append(coeffs[1])

    k1 = pd.Series(k1_vals, index=years)
    k2 = pd.Series(k2_vals, index=years)

    return k1, k2, xbar


def reconstruct_qx_cbd(k1, k2, xbar, ages):
    """Reconstruct qx matrix from CBD parameters."""
    qx = pd.DataFrame(index=ages, columns=k1.index)
    for year in k1.index:
        logit_qx = k1[year] + k2[year] * (ages - xbar)
        qx[year] = np.exp(logit_qx) / (1 + np.exp(logit_qx))
    return qx.astype(float)


def forecast_cbd(k1, k2, n_steps):
    """Forecast k1 and k2 independently with ARIMA."""
    def arima_forecast(series, steps):
        model = pm.auto_arima(
            series.values, seasonal=False,
            stepwise=True, suppress_warnings=True, error_action='ignore'
        )
        fc, ci = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return fc, ci, model

    fc1, ci1, m1 = arima_forecast(k1, n_steps)
    fc2, ci2, m2 = arima_forecast(k2, n_steps)

    last_year = k1.index[-1]
    future_years = range(last_year + 1, last_year + 1 + n_steps)

    k1_fc = pd.Series(fc1, index=future_years)
    k2_fc = pd.Series(fc2, index=future_years)

    return k1_fc, k2_fc, ci1, ci2