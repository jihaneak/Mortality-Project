import numpy as np
import pandas as pd


def compute_ax(df, method="constant"):
    """
    Calcule ax — fraction d'année vécue en moyenne l'année du décès.

    Méthodes
    --------
    'constant'     : ax = 0.5 pour tous les âges (défaut)
    'coale_demeny' : correction à l'âge 0 et 1 selon la mortalité infantile
    """
    ax = 0.5 * np.ones(len(df))

    if method == "coale_demeny":
        m0 = df.loc[df.index[0], "mx"]
        ax[0] = 0.330 if m0 >= 0.107 else 0.045 + 2.684 * m0
        if len(df) > 1:
            ax[1] = 1.5

    return ax


def compute_life_table(df, ax_method="constant"):
    """
    Construit la table de mortalité complète à partir d'un DataFrame Age/mx.

    Colonnes retournées : Age, mx, ax, qx, px, lx, dx, Lx, Tx, ex

    Paramètres
    ----------
    df        : DataFrame avec colonnes 'Age' et 'mx'
    ax_method : 'constant' (défaut) ou 'coale_demeny'
    """
    df = df.reset_index(drop=True).sort_values('Age').copy()

    # ax
    if "ax" not in df.columns or df["ax"].isnull().any():
        df["ax"] = compute_ax(df, method=ax_method)

    # qx = mx / (1 + (1 - ax) * mx)
    if "qx" not in df.columns:
        df["qx"] = df["mx"] / (1 + (1 - df["ax"]) * df["mx"])

    df["qx"] = df["qx"].clip(upper=1.0)
    df["px"] = 1 - df["qx"]

    # lx (radix = 100 000)
    df["lx"] = 0.0
    df.loc[df.index[0], "lx"] = 100_000.0
    for i in range(1, len(df)):
        df.loc[df.index[i], "lx"] = (
            df.loc[df.index[i - 1], "lx"] * df.loc[df.index[i - 1], "px"]
        )

    df["dx"] = df["lx"] * df["qx"]
    df["Lx"] = df["lx"] - (1 - df["ax"]) * df["dx"]

    # Dernier âge : Lx = lx / mx  (hypothèse de fermeture)
    last = df.index[-1]
    if df.loc[last, "mx"] > 0:
        df.loc[last, "Lx"] = df.loc[last, "lx"] / df.loc[last, "mx"]

    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    return df