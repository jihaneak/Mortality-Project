import numpy as np
import pandas as pd


def fit_lee_carter(df, age_col='Age', year_col='Year', mx_col='mx'):
    """
    Calibre le modèle Lee-Carter par SVD.

    Modèle : log(mx(t)) = ax + bx * kt + εxt

    Contraintes d'identifiabilité :
      - sum(bx) = 1
      - sum(kt) = 0  (moyenne absorbée dans ax)

    Retourne
    --------
    ax : pd.Series indexée par âge   — profil moyen de mortalité
    bx : pd.Series indexée par âge   — sensibilité au temps
    kt : pd.Series indexée par année — indice de mortalité
    """
    df_pivot = df.pivot(index=age_col, columns=year_col, values=mx_col)
    log_mx   = np.log(df_pivot.clip(lower=1e-10))
    ages     = log_mx.index
    years    = log_mx.columns

    ax       = log_mx.mean(axis=1)
    centered = log_mx.subtract(ax, axis=0)

    U, s, Vt = np.linalg.svd(centered.values, full_matrices=False)
    bx_raw   = U[:, 0]
    kt_raw   = s[0] * Vt[0, :]

    # Convention de signe : bx positif (mortalité décroît avec le temps)
    if bx_raw.sum() < 0:
        bx_raw = -bx_raw
        kt_raw = -kt_raw

    # Normalisation : sum(bx) = 1
    s_bx   = bx_raw.sum()
    bx_raw = bx_raw / s_bx
    kt_raw = kt_raw * s_bx

    # Centrage : sum(kt) = 0, moyenne absorbée dans ax
    kt_mean = kt_raw.mean()
    kt_raw  = kt_raw - kt_mean
    ax      = ax + bx_raw * kt_mean

    ax = pd.Series(ax.values if hasattr(ax, 'values') else ax, index=ages)
    bx = pd.Series(bx_raw, index=ages)
    kt = pd.Series(kt_raw, index=years)

    return ax, bx, kt