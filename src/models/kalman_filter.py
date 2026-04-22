import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.life_expectancy import compute_life_table


# ── MLE des hyperparamètres ───────────────────────────────────────────────────

def _kalman_loglik(params, kt_obs, drift):
    """Log-vraisemblance négative du modèle d'état-espace (pour minimisation)."""
    sp = np.exp(params[0])
    so = np.exp(params[1])
    mu_f = kt_obs[0]
    P_f  = sp ** 2
    ll   = 0.0
    for t in range(1, len(kt_obs)):
        mu_p = mu_f + drift
        P_p  = P_f  + sp ** 2
        v    = kt_obs[t] - mu_p
        S    = P_p + so ** 2
        ll  -= 0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)
        K    = P_p / S
        mu_f = mu_p + K * v
        P_f  = (1 - K) * P_p
    return -ll


# ── Filtrage ─────────────────────────────────────────────────────────────────

def _run_filter(kt_obs, drift, sigma_proc, sigma_obs):
    """Applique le filtre de Kalman et retourne états filtrés et variances."""
    n     = len(kt_obs)
    mu_f  = np.zeros(n)
    P_f   = np.zeros(n)
    mu_f[0] = kt_obs[0]
    P_f[0]  = sigma_proc ** 2
    for t in range(1, n):
        mu_p    = mu_f[t - 1] + drift
        P_p     = P_f[t - 1]  + sigma_proc ** 2
        S       = P_p + sigma_obs ** 2
        K       = P_p / S
        mu_f[t] = mu_p + K * (kt_obs[t] - mu_p)
        P_f[t]  = (1 - K) * P_p
    return mu_f, P_f


# ── API publique ──────────────────────────────────────────────────────────────

def fit_kalman(ax, bx, kt_train, df_train,
               sigma_obs_scale=2.0):
    """
    Calibre le filtre de Kalman bayésien sur kt.

    Modèle d'état-espace :
      État  : kt = kt-1 + drift + η,   η ~ N(0, σ²_proc)
      Obs   : kt_obs = kt + ε,          ε ~ N(0, σ²_obs)

    σ_obs est multiplié par sigma_obs_scale après MLE pour réduire
    le gain de Kalman et éviter les oscillations (calibration empirique).

    Retourne
    --------
    dict avec clés : mu_filtered, P_filtered, sigma_proc, sigma_obs,
                     sigma_obs_cal, drift, kt_obs_raw, residual_std
    """
    ax_v = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v = bx.values if hasattr(bx, 'values') else np.asarray(bx)

    years_tr     = list(kt_train.index)
    T            = len(years_tr)
    df_pivot     = df_train.pivot(index='Age', columns='Year', values='mx')
    log_mx_train = np.log(df_pivot.clip(lower=1e-10).values)   # (A, T)

    # Résidus Lee-Carter
    log_mx_fit   = np.column_stack([ax_v + bx_v * kt_train[y] for y in years_tr])
    residual_std = float(np.std((log_mx_fit - log_mx_train).ravel()))

    # kt observé bruyant (proxy)
    kt_obs_raw = np.array([
        np.dot(bx_v, log_mx_train[:, t] - ax_v) / np.dot(bx_v, bx_v)
        for t in range(T)
    ])

    drift = float(np.mean(np.diff(kt_obs_raw)))

    # MLE pour σ_proc et σ_obs
    res = minimize(
        _kalman_loglik,
        x0=[np.log(2.0), np.log(1.0)],
        args=(kt_obs_raw, drift),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6}
    )
    sigma_proc    = float(np.exp(res.x[0]))
    sigma_obs_raw = float(np.exp(res.x[1]))
    sigma_obs_cal = sigma_obs_raw * sigma_obs_scale

    # Filtrage sur le training
    mu_filt, P_filt = _run_filter(kt_obs_raw, drift, sigma_proc, sigma_obs_cal)

    return {
        'mu_filtered':   mu_filt,
        'P_filtered':    P_filt,
        'sigma_proc':    sigma_proc,
        'sigma_obs_raw': sigma_obs_raw,
        'sigma_obs_cal': sigma_obs_cal,
        'drift':         drift,
        'kt_obs_raw':    kt_obs_raw,
        'residual_std':  residual_std,
    }


def predict_bayesian(kf_params, ax, bx, n_mc=800):
    """
    Génère des tirages MC de e0 depuis la distribution prédictive.

    Distribution prédictive :
      kt+1 ~ N(mu_pred,  P_pred + σ²_obs_cal)

    Retourne
    --------
    np.array de longueur n_mc des valeurs e0 simulées
    """
    ax_v = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v = bx.values if hasattr(bx, 'values') else np.asarray(bx)
    A    = len(ax_v)
    ages = ax.index.tolist() if hasattr(ax, 'index') else list(range(A))

    mu_pred      = kf_params['mu_cur'] + kf_params['drift']
    P_pred       = max(kf_params['P_cur'] + kf_params['sigma_proc'] ** 2,
                       kf_params['sigma_proc'] ** 2)
    kt_total_std = np.sqrt(P_pred + kf_params['sigma_obs_cal'] ** 2)
    residual_std = kf_params['residual_std']

    kt_s  = np.random.normal(mu_pred, kt_total_std, size=(n_mc, 1))
    noise = np.random.normal(0, residual_std,        size=(n_mc, A))
    mx_s  = np.exp(ax_v + bx_v * kt_s + noise).clip(min=1e-10)

    return np.array([
        compute_life_table(pd.DataFrame({'Age': ages, 'mx': mx_s[i]})).iloc[0]['ex']
        for i in range(n_mc)
    ]), mu_pred, P_pred


def rolling_backtest_bayesian(ax, bx, kt_train, kf_fit, df_test,
                               n_mc=800, age_max=90):
    """
    Backtest rolling bayésien (Kalman Filter) sur Lee-Carter.

    Retourne
    --------
    dict avec clés : years, e0_obs, e0_pred, e0_lower, e0_upper,
                     rmse, bias, coverage
    """
    ax_v = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v = bx.values if hasattr(bx, 'values') else np.asarray(bx)

    sigma_proc    = kf_fit['sigma_proc']
    sigma_obs_cal = kf_fit['sigma_obs_cal']
    drift         = kf_fit['drift']
    residual_std  = kf_fit['residual_std']
    P_MIN         = sigma_proc ** 2

    years_test = sorted(df_test['Year'].unique())
    mu_cur     = float(kf_fit['mu_filtered'][-1])
    P_cur      = float(kf_fit['P_filtered'][-1])

    e0_obs, e0_pred, e0_lower, e0_upper = [], [], [], []

    for year in years_test:

        df_yr_obs  = df_test[
            (df_test['Year'] == year) & (df_test['Age'] <= age_max)
        ][['Age', 'mx']].sort_values('Age').reset_index(drop=True)
        log_mx_obs = np.log(df_yr_obs['mx'].clip(lower=1e-10).values)
        e0_true    = compute_life_table(df_yr_obs).iloc[0]['ex']
        e0_obs.append(e0_true)

        # Prédiction
        mu_pred = mu_cur + drift
        P_pred  = max(P_cur + sigma_proc ** 2, P_MIN)

        # MC e0
        kf_state = {
            'mu_cur':        mu_cur,
            'P_cur':         P_cur,
            'drift':         drift,
            'sigma_proc':    sigma_proc,
            'sigma_obs_cal': sigma_obs_cal,
            'residual_std':  residual_std,
        }
        e0_mc, _, _ = predict_bayesian(kf_state, ax, bx, n_mc=n_mc)
        e0_pred.append(float(np.mean(e0_mc)))
        e0_lower.append(float(np.percentile(e0_mc, 2.5)))
        e0_upper.append(float(np.percentile(e0_mc, 97.5)))

        # Mise à jour bayésienne
        kt_new = np.dot(bx_v, log_mx_obs - ax_v) / np.dot(bx_v, bx_v)
        S      = P_pred + sigma_obs_cal ** 2
        K      = P_pred / S
        mu_cur = mu_pred + K * (kt_new - mu_pred)
        P_cur  = (1 - K) * P_pred

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