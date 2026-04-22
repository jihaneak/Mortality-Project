import numpy as np
import pandas as pd
import pmdarima as pm

from src.models.life_expectancy import compute_life_table


# ── Calcul de la rente ────────────────────────────────────────────────────────

def compute_annuity(df_proj, age_x, taux, horizon=40):
    """
    Calcule äx — valeur actualisée espérée d'une rente viagère immédiate.

    äx = Σ_{k=0}^{ω} v^k · k_px

    avec v = 1/(1+i),  k_px = lx+k / lx

    Paramètres
    ----------
    df_proj : DataFrame Age/mx projeté (ages 0 à 90+)
    age_x   : âge de souscription
    taux    : taux technique annuel (ex: 0.02)
    horizon : nombre max d'années de rente (défaut 40 → jusqu'à 105 ans)

    Retourne
    --------
    äx (float), ou np.nan si l'âge est absent de la table
    """
    lt    = compute_life_table(df_proj).set_index('Age')
    v     = 1 / (1 + taux)
    lx_ref = lt.loc[age_x, 'lx'] if age_x in lt.index else None

    if lx_ref is None or lx_ref < 1e-10:
        return np.nan

    ax_val = 0.0
    for k in range(horizon + 1):
        age_k = age_x + k
        if age_k not in lt.index:
            break
        ax_val += (v ** k) * (lt.loc[age_k, 'lx'] / lx_ref)

    return float(ax_val)


def annual_premium(annuity_val, capital):
    """
    Prime annuelle d'une rente viagère immédiate.

    Prime = Capital / äx
    """
    return float(capital / annuity_val) if annuity_val > 0 else np.nan


# ── VaR de longévité ─────────────────────────────────────────────────────────

def annuity_var_cbd(k1_train, k2_train, ages_cbd, xbar, cbd_residual_std,
                    df_lc_base, age_x, taux, n_forecast,
                    capital=100_000, n_boot=1000):
    """
    Distribution bootstrap de äx via CBD — quantifie le risque de longévité.

    La dispersion est propagée depuis l'incertitude sur k1, k2 (marche
    aléatoire) et le résidu du modèle CBD.

    Paramètres
    ----------
    k1_train, k2_train : pd.Series kt calibrés sur training
    ages_cbd           : np.array des âges CBD (50–85)
    xbar               : âge moyen CBD
    cbd_residual_std   : écart-type résidus logit(qx)
    df_lc_base         : table Lee-Carter projetée (ages 0–90) — base pour
                         les âges hors zone CBD
    age_x              : âge de souscription
    taux               : taux technique
    n_forecast         : nombre d'années à projeter (ex: 25 pour 2025)
    capital            : montant converti en rente
    n_boot             : simulations bootstrap

    Retourne
    --------
    dict avec clés : ax_central, ax_p5, ax_p95,
                     prime_central, prime_p5, prime_p95,
                     longevity_risk
    """
    m_k1    = pm.auto_arima(k1_train.values, seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action='ignore')
    m_k2    = pm.auto_arima(k2_train.values, seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action='ignore')
    k1_pt   = float(m_k1.predict(n_periods=n_forecast)[-1])
    k2_pt   = float(m_k2.predict(n_periods=n_forecast)[-1])
    k1_std  = float(np.std(np.diff(k1_train.values))) * np.sqrt(n_forecast)
    k2_std  = float(np.std(np.diff(k2_train.values))) * np.sqrt(n_forecast)

    # äx central
    logit_q  = k1_pt + k2_pt * (ages_cbd - xbar)
    qx_c     = (np.exp(logit_q) / (1 + np.exp(logit_q))).clip(1e-6, 1 - 1e-6)
    mx_c     = (qx_c / (1 - 0.5 * qx_c)).clip(min=1e-10)
    df_c     = df_lc_base.copy()
    for i, age in enumerate(ages_cbd.astype(int)):
        df_c.loc[df_c['Age'] == age, 'mx'] = mx_c[i]
    ax_central = compute_annuity(df_c, age_x, taux)

    # Bootstrap
    ax_boot = []
    for _ in range(n_boot):
        k1_s    = np.random.normal(k1_pt, k1_std)
        k2_s    = np.random.normal(k2_pt, k2_std)
        noise   = np.random.normal(0, cbd_residual_std, size=len(ages_cbd))
        logit_s = k1_s + k2_s * (ages_cbd - xbar) + noise
        qx_s    = (np.exp(logit_s) / (1 + np.exp(logit_s))).clip(1e-6, 1 - 1e-6)
        mx_s    = (qx_s / (1 - 0.5 * qx_s)).clip(min=1e-10)
        df_s    = df_lc_base.copy()
        for i, age in enumerate(ages_cbd.astype(int)):
            df_s.loc[df_s['Age'] == age, 'mx'] = mx_s[i]
        ax_boot.append(compute_annuity(df_s, age_x, taux))

    ax_boot = np.array([x for x in ax_boot if not np.isnan(x)])
    p5  = float(np.percentile(ax_boot, 5))
    p95 = float(np.percentile(ax_boot, 95))

    return {
        'ax_central':      ax_central,
        'ax_p5':           p5,
        'ax_p95':          p95,
        'prime_central':   annual_premium(ax_central, capital),
        'prime_p5':        annual_premium(p95, capital),   # äx élevé → prime basse
        'prime_p95':       annual_premium(p5,  capital),   # äx faible → prime haute
        'longevity_risk':  p95 - p5,
    }


# ── Pricing multi-modèles ─────────────────────────────────────────────────────

def price_all_models(models_mx, age_x, taux, capital=100_000):
    """
    Calcule äx et la prime annuelle pour un dict de tables projetées.

    Paramètres
    ----------
    models_mx : dict {nom_modèle: DataFrame Age/mx projeté}
    age_x     : âge de souscription
    taux      : taux technique
    capital   : montant converti

    Retourne
    --------
    pd.DataFrame avec colonnes : Modèle, äx, Prime (€/an)
    """
    rows = []
    for name, df_proj in models_mx.items():
        ax_val = compute_annuity(df_proj, age_x, taux)
        rows.append({
            'Modèle':      name,
            'äx':          round(ax_val, 4) if not np.isnan(ax_val) else np.nan,
            'Prime (€/an)': round(annual_premium(ax_val, capital)) if not np.isnan(ax_val) else np.nan,
        })
    return pd.DataFrame(rows).set_index('Modèle')