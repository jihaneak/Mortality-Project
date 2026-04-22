import numpy as np
import pandas as pd
import pmdarima as pm

from src.models.life_expectancy import compute_life_table
from src.models.uncertainty     import bootstrap_e0_ci


def extract_kt(ax, bx, df_year):
    """
    Ré-extrait kt pour une année via projection moindres carrés.

    kt = (bx · (log(mx) - ax)) / (bx · bx)
    """
    ax_v       = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v       = bx.values if hasattr(bx, 'values') else np.asarray(bx)
    log_mx_obs = np.log(df_year['mx'].clip(lower=1e-10).values)
    return float(np.dot(bx_v, log_mx_obs - ax_v) / np.dot(bx_v, bx_v))


def compute_residual_std(ax, bx, kt_train, df_train):
    """
    Calcule l'écart-type des résidus Lee-Carter sur le training.

    residual_std = std(log(mx_fit) - log(mx_obs))
    """
    ax_v = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v = bx.values if hasattr(bx, 'values') else np.asarray(bx)

    df_pivot  = df_train.pivot(index='Age', columns='Year', values='mx')
    log_mx_obs = np.log(df_pivot.clip(lower=1e-10).values)
    log_mx_fit = np.column_stack([ax_v + bx_v * kt_train[y] for y in kt_train.index])

    return float(np.std((log_mx_fit - log_mx_obs).ravel()))


def rolling_backtest(ax, bx, kt_train, df_train, df_test,
                     n_boot=300, age_max=90):
    """
    Backtest rolling one-step-ahead sur Lee-Carter.

    Paramètres
    ----------
    ax, bx      : paramètres Lee-Carter (pd.Series indexées par âge)
    kt_train    : pd.Series kt calibré sur training
    df_train    : DataFrame training (pour residual_std)
    df_test     : DataFrame test
    n_boot      : tirages bootstrap pour l'IC
    age_max     : âge max à conserver dans df_test

    Retourne
    --------
    dict avec clés : years, e0_obs, e0_pred, e0_lower, e0_upper,
                     rmse, bias, coverage
    """
    residual_std = compute_residual_std(ax, bx, kt_train, df_train)
    years_test   = sorted(df_test["Year"].unique())
    kt_all       = kt_train.copy()

    e0_obs, e0_pred, e0_lower, e0_upper = [], [], [], []

    for year in years_test:
        # ── Forecast kt (ARIMA rolling) ──────────────────────────────────────
        model    = pm.auto_arima(kt_all.values, seasonal=False, stepwise=True,
                                  suppress_warnings=True, error_action='ignore')
        kt_fc, ci = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        kt_point  = float(kt_fc[0])
        kt_std    = (float(ci[0, 1]) - float(ci[0, 0])) / (2 * 1.96)

        # ── e0 observé ───────────────────────────────────────────────────────
        data_obs = df_test[
            (df_test["Year"] == year) & (df_test["Age"] <= age_max)
        ][['Age', 'mx']].reset_index(drop=True)
        e0_obs.append(compute_life_table(data_obs).iloc[0]['ex'])

        # ── e0 prédit ────────────────────────────────────────────────────────
        ax_v = ax.values if hasattr(ax, 'values') else np.asarray(ax)
        bx_v = bx.values if hasattr(bx, 'values') else np.asarray(bx)
        mx_p = np.exp(ax_v + bx_v * kt_point).clip(min=1e-10)
        df_p = pd.DataFrame({'Age': ax.index.tolist(), 'mx': mx_p})
        e0_pred.append(compute_life_table(df_p).iloc[0]['ex'])

        # ── Bootstrap CI ─────────────────────────────────────────────────────
        lo, hi = bootstrap_e0_ci(
            ax, bx, kt_point, kt_std,
            residual_std   = residual_std,
            kt_train_len   = len(kt_all),
            n_boot         = n_boot,
        )
        e0_lower.append(lo)
        e0_upper.append(hi)

        # ── Update kt_all avec kt observé ────────────────────────────────────
        kt_actual = extract_kt(
            ax, bx,
            df_test[(df_test["Year"] == year) & (df_test["Age"] <= age_max)
                    ][['Age', 'mx']].reset_index(drop=True)
        )
        kt_all = pd.concat([kt_all, pd.Series([kt_actual], index=[year])])

    e0_obs   = np.array(e0_obs)
    e0_pred  = np.array(e0_pred)
    e0_lower = np.array(e0_lower)
    e0_upper = np.array(e0_upper)

    rmse     = float(np.sqrt(np.mean((e0_pred - e0_obs) ** 2)))
    bias     = float(np.mean(e0_pred - e0_obs))
    coverage = float(np.mean((e0_obs >= e0_lower) & (e0_obs <= e0_upper)))

    return {
        "years":    years_test,
        "e0_obs":   e0_obs,
        "e0_pred":  e0_pred,
        "e0_lower": e0_lower,
        "e0_upper": e0_upper,
        "rmse":     rmse,
        "bias":     bias,
        "coverage": coverage,
    }