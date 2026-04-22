import numpy as np
import pandas as pd
from src.models.life_expectancy import compute_life_table


def bootstrap_e0_ci(ax, bx, kt_point, kt_std, residual_std,
                    kt_train_len, n_boot=300, alpha=0.05):
    """
    Intervalle de confiance bootstrap sur e0 — 3 sources d'incertitude.

    Sources
    -------
    1. Incertitude sur kt  : kt ~ N(kt_point, kt_std²)
    2. Résidu log(mx)      : bruit résiduel du modèle Lee-Carter
    3. Incertitude sur bx  : erreur standard estimée depuis les résidus

    Implémentation vectorisée (tous les tirages en une seule opération numpy).

    Paramètres
    ----------
    ax, bx         : pd.Series indexées par âge
    kt_point       : prévision ponctuelle de kt (ARIMA)
    kt_std         : écart-type de la prévision kt (depuis IC ARIMA)
    residual_std   : écart-type des résidus log(mx) sur le training
    kt_train_len   : longueur de l'historique kt (pour estimer bx_se)
    n_boot         : nombre de tirages bootstrap
    alpha          : niveau pour l'IC (0.05 → IC 95%)

    Retourne
    --------
    (e0_lower, e0_upper) : bornes de l'IC
    """
    ax_v   = ax.values if hasattr(ax, 'values') else np.asarray(ax)
    bx_v   = bx.values if hasattr(bx, 'values') else np.asarray(bx)
    n_ages = len(ax_v)
    ages_l = ax.index.tolist() if hasattr(ax, 'index') else list(range(n_ages))

    # Erreur standard de bx estimée depuis les résidus (non arbitraire)
    bx_se = np.abs(
        residual_std / (np.sqrt(kt_train_len) * np.abs(bx_v).clip(min=1e-6))
    ).clip(max=float(np.std(bx_v)) * 0.15)

    # Tirages vectorisés (n_boot, n_ages)
    kt_s   = np.random.normal(kt_point, kt_std,        size=(n_boot, 1))
    noise  = np.random.normal(0,        residual_std,   size=(n_boot, n_ages))
    bx_s   = np.abs(bx_v + np.random.normal(0, bx_se,  size=(n_boot, n_ages)))

    log_mx = ax_v + bx_s * kt_s + noise         # (n_boot, n_ages)
    mx_s   = np.exp(log_mx).clip(min=1e-10)

    e0_samples = []
    for i in range(n_boot):
        df_s = pd.DataFrame({'Age': ages_l, 'mx': mx_s[i]})
        e0_samples.append(compute_life_table(df_s).iloc[0]['ex'])

    e0_arr = np.array(e0_samples)
    return (
        float(np.percentile(e0_arr, 100 * alpha / 2)),
        float(np.percentile(e0_arr, 100 * (1 - alpha / 2))),
    )