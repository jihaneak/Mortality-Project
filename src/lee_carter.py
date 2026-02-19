import numpy as np
import pandas as pd


def prepare_mx_matrix(df, sex="Female", age_min=0, age_max=90):
    """
    Create log-mortality matrix for Lee-Carter.
    Returns:
        log_mx (DataFrame): rows=Age, cols=Year
    """

    df_sub = df[
        (df["Sex"] == sex) &
        (df["Age"] >= age_min) &
        (df["Age"] <= age_max)
    ].copy()

    pivot = df_sub.pivot(
        index="Age",
        columns="Year",
        values="mx"
    )

    log_mx = np.log(pivot)

    return log_mx

def fit_lee_carter(log_mx):

    ax = log_mx.mean(axis=1)
    M_centered = log_mx.sub(ax, axis=0)

    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

    bx = U[:, 0]
    kt = S[0] * Vt[0, :]

    # Identification
    bx = bx / bx.sum()
    kt = kt * bx.sum()
    kt = kt - kt.mean()

    # Demographic sign convention
    if kt[0] < kt[-1]:
        bx = -bx
        kt = -kt

    return ax, bx, kt
