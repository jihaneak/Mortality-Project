import numpy as np


def poisson_mx(df):
    """
    Estime mx et sa variance sous modèle Poisson.

    df doit contenir les colonnes 'Deaths' et 'Exposure'.

    Retourne
    --------
    df avec colonnes ajoutées : mx, var_mx, se_mx
    """
    df = df.copy()
    df['mx']     = df['Deaths'] / df['Exposure']
    df['var_mx'] = df['mx'] / df['Exposure']    # Var(mx) = mx / E  sous Poisson
    df['se_mx']  = np.sqrt(df['var_mx'])
    return df


def delta_qx(df, ax_col='ax'):
    """
    Applique la méthode Delta pour obtenir un IC sur qx à partir de mx.

    qx = mx / (1 + (1 - ax) * mx)
    Var(qx) ≈ (∂qx/∂mx)² * Var(mx)

    df doit contenir 'mx', 'var_mx' et ax_col.

    Retourne
    --------
    df avec colonnes ajoutées : qx, var_qx, se_qx, qx_lower, qx_upper
    """
    df = df.copy()
    denom        = 1 + (1 - df[ax_col]) * df['mx']
    df['qx']     = (df['mx'] / denom).clip(upper=1.0)
    df['var_qx'] = (1 / denom ** 2) ** 2 * df['var_mx']
    df['se_qx']  = np.sqrt(df['var_qx'])
    df['qx_lower'] = (df['qx'] - 1.96 * df['se_qx']).clip(lower=0)
    df['qx_upper'] = (df['qx'] + 1.96 * df['se_qx']).clip(upper=1)
    return df