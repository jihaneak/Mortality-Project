import numpy as np
import pandas as pd
import pmdarima as pm

from src.models.life_expectancy import compute_life_table


# ── Calibration ───────────────────────────────────────────────────────────────

def fit_renshaw_haberman(df_train, age_min=0, age_max=90, cohort_min_obs=5):
    """
    Modèle Renshaw-Haberman (Lee-Carter + effet cohorte γc).

    Modèle : log(mx(t)) = ax + bx·kt + γ(t-x) + εxt

    Implémentation en 2 étapes stables :
      1. Lee-Carter par SVD → ax, bx, kt
      2. Résidus → γc par moyenne cohortale

    Paramètres
    ----------
    df_train       : DataFrame avec colonnes Age, Year, mx
    age_min/max    : plage d'âges (défaut 0–90)
    cohort_min_obs : nb minimum d'observations pour estimer γc

    Retourne
    --------
    dict avec clés : ax, bx, kt, gamma_c, ages_arr, years_arr,
                     residual_std, lc_residual_std
    """
    df_m      = df_train[(df_train['Age'] >= age_min) & (df_train['Age'] <= age_max)].copy()
    years_tr  = sorted(df_m['Year'].unique())
    ages_all  = sorted(df_m['Age'].unique())
    ages_arr  = np.array(ages_all,  dtype=float)
    years_arr = np.array(years_tr,  dtype=float)
    A, T      = len(ages_arr), len(years_arr)

    df_pivot  = df_m.pivot(index='Age', columns='Year', values='mx')
    log_mx    = np.log(df_pivot.values.clip(min=1e-10))   # (A, T)

    # ── Étape 1 : Lee-Carter ─────────────────────────────────────────────────
    ax = log_mx.mean(axis=1)
    U, S, Vt = np.linalg.svd(log_mx - ax[:, None], full_matrices=False)
    bx = U[:, 0]; kt = S[0] * Vt[0, :]

    s     = bx.sum(); bx /= s; kt *= s
    shift = kt.mean(); kt -= shift; ax += bx * shift

    lc_residual_std = float(np.std((log_mx - ax[:, None] - bx[:, None] * kt[None, :]).ravel()))

    # ── Étape 2 : résidus → γc cohortale ─────────────────────────────────────
    R = log_mx - ax[:, None] - bx[:, None] * kt[None, :]   # (A, T)

    cohort_vals = {}
    for ti, t in enumerate(years_arr):
        for ai, x in enumerate(ages_arr):
            cohort_vals.setdefault(t - x, []).append(R[ai, ti])

    gamma_c = {
        c: float(np.mean(v))
        for c, v in cohort_vals.items()
        if len(v) >= cohort_min_obs
    }

    # Résidu std après correction cohortale
    R_after = R.copy()
    for ti, t in enumerate(years_arr):
        for ai, x in enumerate(ages_arr):
            c = t - x
            if c in gamma_c:
                R_after[ai, ti] -= gamma_c[c]

    residual_std = float(np.std(R_after.ravel()))

    return {
        'ax':               ax,
        'bx':               bx,
        'kt':               pd.Series(kt, index=years_tr),
        'gamma_c':          gamma_c,
        'ages_arr':         ages_arr,
        'years_arr':        years_arr,
        'residual_std':     residual_std,
        'lc_residual_std':  lc_residual_std,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_gamma(cohort, gamma_c, gamma_series=None):
    """
    Retourne γc pour une cohorte donnée.
    Si inconnue → moyenne des 5 dernières cohortes observées.
    """
    if cohort in gamma_c:
        return gamma_c[cohort]
    if gamma_series is None:
        gamma_series = pd.Series(gamma_c).sort_index()
    return float(gamma_series.iloc[-5:].mean())


def predict_rh(rh_params, kt_point, year, gamma_series=None):
    """
    Reconstruit mx prédit pour une année avec RH.

    Retourne DataFrame Age/mx.
    """
    ax        = rh_params['ax']
    bx        = rh_params['bx']
    ages_arr  = rh_params['ages_arr']
    gamma_c   = rh_params['gamma_c']

    if gamma_series is None:
        gamma_series = pd.Series(gamma_c).sort_index()

    log_mx_pred = np.array([
        ax[ai] + bx[ai] * kt_point
        + get_gamma(year - ages_arr[ai], gamma_c, gamma_series)
        for ai in range(len(ages_arr))
    ])
    mx_pred = np.exp(log_mx_pred).clip(min=1e-10)
    return pd.DataFrame({'Age': ages_arr.astype(int), 'mx': mx_pred})


# ── Backtest ──────────────────────────────────────────────────────────────────

def rolling_backtest_rh(rh_params, df_test, n_boot=200, age_max=90):
    """
    Backtest rolling one-step-ahead pour Renshaw-Haberman.

    Retourne
    --------
    dict avec clés : years, e0_obs, e0_pred, e0_lower, e0_upper,
                     rmse, bias, coverage
    """
    ax           = rh_params['ax']
    bx           = rh_params['bx']
    gamma_c      = rh_params['gamma_c']
    ages_arr     = rh_params['ages_arr']
    residual_std = rh_params['residual_std']
    A            = len(ages_arr)

    gamma_series = pd.Series(gamma_c).sort_index()
    kt_all       = rh_params['kt'].copy()
    years_test   = sorted(df_test['Year'].unique())

    e0_obs, e0_pred, e0_lower, e0_upper = [], [], [], []

    for year in years_test:

        df_yr_obs = df_test[
            (df_test['Year'] == year) & (df_test['Age'] <= age_max)
        ][['Age', 'mx']].sort_values('Age').reset_index(drop=True)

        e0_obs.append(compute_life_table(df_yr_obs).iloc[0]['ex'])

        # Forecast kt
        m_kt = pm.auto_arima(kt_all.values, seasonal=False, stepwise=True,
                              suppress_warnings=True, error_action='ignore')
        fc_kt, ci_kt = m_kt.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        kt_pt  = float(fc_kt[0])
        kt_std = (float(ci_kt[0, 1]) - float(ci_kt[0, 0])) / (2 * 1.96)

        # e0 prédit
        df_pred = predict_rh(rh_params, kt_pt, year, gamma_series)
        e0_pred.append(compute_life_table(df_pred).iloc[0]['ex'])

        # Bootstrap CI
        e0_boot = []
        for _ in range(n_boot):
            kt_s  = np.random.normal(kt_pt, kt_std)
            ns    = np.random.normal(0, residual_std, size=A)
            log_s = np.array([
                ax[ai] + bx[ai] * kt_s
                + get_gamma(year - ages_arr[ai], gamma_c, gamma_series)
                + ns[ai]
                for ai in range(A)
            ])
            mx_s = np.exp(log_s).clip(min=1e-10)
            df_s = pd.DataFrame({'Age': ages_arr.astype(int), 'mx': mx_s})
            e0_boot.append(compute_life_table(df_s).iloc[0]['ex'])

        e0_lower.append(float(np.percentile(e0_boot, 2.5)))
        e0_upper.append(float(np.percentile(e0_boot, 97.5)))

        # Update kt_all avec l'observé
        log_mx_obs  = np.log(df_yr_obs['mx'].values.clip(min=1e-10))
        gamma_obs   = np.array([get_gamma(year - x, gamma_c, gamma_series)
                                 for x in ages_arr])
        kt_obs      = float(np.dot(bx, log_mx_obs - ax - gamma_obs)
                            / np.dot(bx, bx))
        kt_all      = pd.concat([kt_all, pd.Series([kt_obs], index=[year])])

    e0_obs   = np.array(e0_obs);   e0_pred  = np.array(e0_pred)
    e0_lower = np.array(e0_lower); e0_upper = np.array(e0_upper)

    return {
        'years':    years_test,
        'e0_obs':   e0_obs,
        'e0_pred':  e0_pred,
        'e0_lower': e0_lower,
        'e0_upper': e0_upper,
        'rmse':     float(np.sqrt(np.mean((e0_pred - e0_obs) ** 2))),
        'bias':     float(np.mean(e0_pred - e0_obs)),
        'coverage': float(np.mean((e0_obs >= e0_lower) & (e0_obs <= e0_upper))),
    }