import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def fit_lee_carter(df, age_col='Age', year_col='Year', mx_col='mx'):
    """
    Calibre Lee-Carter: log(mx) = ax + bx * kt
    Retourne ax, bx, kt pour une population donnée.
    """
    df_fit = df[df[age_col] <= 90].copy()
    df_pivot = df_fit.pivot(index=age_col, columns=year_col, values=mx_col)
    
    log_mx = np.log(df_pivot.clip(lower=1e-10))
    ages = log_mx.index
    years = log_mx.columns

    ax = log_mx.mean(axis=1)
    centered = log_mx.subtract(ax, axis=0)

    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    bx = U[:, 0]
    kt = s[0] * Vt[0, :]

    if np.sum(bx) < 0:
        bx = -bx
        kt = -kt
    
    sum_bx = np.sum(bx)
    bx = bx / sum_bx
    kt = kt * sum_bx

    ax = pd.Series(ax, index=ages)
    bx = pd.Series(bx, index=ages)
    
    return ax, bx, kt

def forecast_kt(kt, n_years=20):
    """
    Projection de kt avec calcul explicite du drift (dérive).
    """
    kt = np.asarray(kt)
    
    drift = (kt[-1] - kt[0]) / (len(kt) - 1)
    
    steps = np.arange(1, n_years + 1)
    kt_future = kt[-1] + (drift * steps)
    
    return pd.Series(kt_future)

def reconstruct_mx(ax, bx, kt_future):
    """
    Reconstruit les mx futurs. 
    Fonctionne avec des Series Pandas ou des Numpy Arrays.
    """
    ax_val = np.asarray(ax)
    bx_val = np.asarray(bx)
    kt_val = np.asarray(kt_future)
    
    log_mx_future = ax_val[:, None] + np.outer(bx_val, kt_val)
    
    ages = ax.index if hasattr(ax, 'index') else range(len(ax))
    years = range(1, len(kt_val) + 1)
    
    return pd.DataFrame(np.exp(log_mx_future), index=ages, columns=years)