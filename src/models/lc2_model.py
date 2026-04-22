import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

from src.models.life_expectancy import compute_life_table


def fit_lc2(df, age_col='Age', year_col='Year', mx_col='mx'):
    """
    Lee-Carter à 2 facteurs.

    Modèle : log(mx) = ax + bx1·kt1 + bx2·kt2 + εxt

    kt1 capte le trend global de mortalité (baisse long-terme).
    kt2 capte les patterns résiduels par âge (chocs spécifiques).

    Particulièrement adapté à la mortalité masculine qui présente
    une dualité structurelle : chocs 20–45 ans + trend 60–90 ans.

    Retourne
    --------
    ax, bx1, bx2 : pd.Series indexées par âge
    kt1, kt2     : pd.Series indexées par année
    """
    df_pivot = df.pivot(index=age_col, columns=year_col, values=mx_col)
    log_mx   = np.log(df_pivot.clip(lower=1e-10))
    ages     = log_mx.index
    years    = log_mx.columns

    ax       = log_mx.mean(axis=1)
    centered = log_mx.subtract(ax, axis=0)

    U, s, Vt = np.linalg.svd(centered.values, full_matrices=False)

    # ── Composant 1 ───────────────────────────────────────────────────────────
    bx1_raw = U[:, 0]; kt1_raw = s[0] * Vt[0, :]
    if bx1_raw.sum() < 0:
        bx1_raw, kt1_raw = -bx1_raw, -kt1_raw
    s1 = bx1_raw.sum(); bx1_raw /= s1; kt1_raw *= s1
    shift1 = kt1_raw.mean(); kt1_raw -= shift1; ax = ax + bx1_raw * shift1

    # ── Composant 2 ───────────────────────────────────────────────────────────
    bx2_raw = U[:, 1]; kt2_raw = s[1] * Vt[1, :]
    if bx2_raw.sum() < 0:
        bx2_raw, kt2_raw = -bx2_raw, -kt2_raw
    s2 = bx2_raw.sum(); bx2_raw /= s2; kt2_raw *= s2
    shift2 = kt2_raw.mean(); kt2_raw -= shift2; ax = ax + bx2_raw * shift2

    var_total = np.sum(s ** 2)
    var1 = s[0] ** 2 / var_total * 100
    var2 = s[1] ** 2 / var_total * 100

    print(f'  LC2 — variance expliquée : kt1={var1:.1f}%  kt2={var2:.1f}%  '
          f'total={var1+var2:.1f}%')

    return (
        pd.Series(ax.values if hasattr(ax, 'values') else ax, index=ages),
        pd.Series(bx1_raw, index=ages),
        pd.Series(bx2_raw, index=ages),
        pd.Series(kt1_raw, index=years),
        pd.Series(kt2_raw, index=years),
    )


def compute_residual_std_lc2(ax, bx1, bx2, kt1, kt2, df_train):
    """Résidu std du modèle LC2 sur le training."""
    pivot    = df_train.pivot(index='Age', columns='Year', values='mx')
    log_obs  = np.log(pivot.clip(lower=1e-10).values)
    log_fit2 = np.column_stack([
        ax.values + bx1.values * kt1[y] + bx2.values * kt2[y]
        for y in kt1.index
    ])
    return float(np.std((log_fit2 - log_obs).ravel()))


def extract_kt_lc2(ax, bx1, bx2, df_year):
    """Extrait kt1 et kt2 pour une année via OLS."""
    log_obs = np.log(df_year['mx'].clip(lower=1e-10).values)
    y = log_obs - ax.values
    X = np.column_stack([bx1.values, bx2.values])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coeffs[0]), float(coeffs[1])


def _fast_forecast(series, order):
    """Forecast 1 pas avec ordre ARIMA fixé."""
    try:
        m   = StatsARIMA(series, order=order).fit(disp=False)
        pt  = float(m.forecast(1).iloc[0])
        std = float(np.sqrt(m.params.get('sigma2', m.resid.var())))
        return pt, std
    except:
        drift = float(np.mean(np.diff(series)))
        return float(series[-1]) + drift, float(np.std(np.diff(series)))


def rolling_backtest_lc2(ax, bx1, bx2, kt1, kt2, df_train, df_test,
                          n_boot=200, age_max=90):
    """
    Backtest rolling one-step-ahead pour LC2.

    Retourne dict avec years/e0_obs/e0_pred/e0_lower/e0_upper/rmse/bias/coverage
    """
    res_std    = compute_residual_std_lc2(ax, bx1, bx2, kt1, kt2, df_train)
    years_test = sorted(df_test['Year'].unique())

    # Ordre ARIMA fixé une seule fois
    m1 = pm.auto_arima(kt1.values, seasonal=False, stepwise=True,
                        suppress_warnings=True, error_action='ignore')
    m2 = pm.auto_arima(kt2.values, seasonal=False, stepwise=True,
                        suppress_warnings=True, error_action='ignore')
    order1 = m1.order; order2 = m2.order

    kt1_all = kt1.copy(); kt2_all = kt2.copy()
    e0_obs, e0_pred, e0_lo, e0_hi = [], [], [], []
    ages = ax.index.tolist()
    A    = len(ages)

    for year in years_test:
        df_yr = df_test[
            (df_test['Year'] == year) & (df_test['Age'] <= age_max)
        ][['Age', 'mx']].sort_values('Age').reset_index(drop=True)

        e0_obs.append(compute_life_table(df_yr).iloc[0]['ex'])

        kt1_pt, kt1_std = _fast_forecast(kt1_all.values, order1)
        kt2_pt, kt2_std = _fast_forecast(kt2_all.values, order2)

        # Point forecast
        mx_p = np.exp(ax.values + bx1.values*kt1_pt + bx2.values*kt2_pt).clip(min=1e-10)
        e0_pred.append(compute_life_table(
            pd.DataFrame({'Age': ages, 'mx': mx_p})
        ).iloc[0]['ex'])

        # Bootstrap vectorisé
        kt1_s = np.random.normal(kt1_pt, kt1_std, size=(n_boot, 1))
        kt2_s = np.random.normal(kt2_pt, kt2_std, size=(n_boot, 1))
        noise = np.random.normal(0, res_std,       size=(n_boot, A))

        log_s = ax.values + bx1.values*kt1_s + bx2.values*kt2_s + noise
        mx_s  = np.exp(log_s).clip(min=1e-10)

        e0_boot = [
            compute_life_table(pd.DataFrame({'Age': ages, 'mx': mx_s[i]})).iloc[0]['ex']
            for i in range(n_boot)
        ]
        e0_lo.append(float(np.percentile(e0_boot, 2.5)))
        e0_hi.append(float(np.percentile(e0_boot, 97.5)))

        # Update kt1/kt2
        kt1_obs, kt2_obs = extract_kt_lc2(ax, bx1, bx2, df_yr)
        kt1_all = pd.concat([kt1_all, pd.Series([kt1_obs], index=[year])])
        kt2_all = pd.concat([kt2_all, pd.Series([kt2_obs], index=[year])])

    e0_obs  = np.array(e0_obs);  e0_pred = np.array(e0_pred)
    e0_lo   = np.array(e0_lo);   e0_hi   = np.array(e0_hi)
    rmse    = float(np.sqrt(np.mean((e0_pred - e0_obs) ** 2)))
    bias    = float(np.mean(e0_pred - e0_obs))
    coverage = float(np.mean((e0_obs >= e0_lo) & (e0_obs <= e0_hi)))

    return {
        'years':    years_test,
        'e0_obs':   e0_obs,
        'e0_pred':  e0_pred,
        'e0_lower': e0_lo,
        'e0_upper': e0_hi,
        'rmse':     rmse,
        'bias':     bias,
        'coverage': coverage,
        'residual_std': res_std,
    }