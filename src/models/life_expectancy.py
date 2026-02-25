import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

def compute_ax(df, method="constant"):
    """
    Calcule ax selon la méthode choisie pour ajuster le temps vécu 
    au cours de l'année de décès (essentiel pour les jeunes âges).
    """
    ax = 0.5 * np.ones(len(df))
    
    if method == "data_based":
        ax = 1 / (2 + df["mx"])
    
    elif method == "coale_demeny":
        ax = 0.5 * np.ones(len(df))
        m0 = df.loc[df.index[0], "mx"]
        if m0 >= 0.107:
            ax[0] = 0.330
        else:
            ax[0] = 0.045 + 2.684 * m0
        
        if len(df) > 1:
            ax[1] = 1.5
    
    return ax

def compute_life_table(df, ax_method="constant"):
    """
    Construction complète de la table de mortalité.
    ax_method: 'constant', 'data_based', ou 'coale_demeny'
    """
    df = df.reset_index(drop=True).sort_values('Age').copy()
    
    if "ax" not in df.columns or df["ax"].isnull().any():
        df["ax"] = compute_ax(df, method=ax_method)
    
    if "qx" not in df.columns:
        df["qx"] = df["mx"] / (1 + (1 - df["ax"]) * df["mx"])
    
    df['px'] = 1 - df['qx']
    
    df['lx'] = 0.0
    df.loc[df.index[0], 'lx'] = 100000
    for i in range(1, len(df)):
        df.loc[df.index[i], 'lx'] = df.loc[df.index[i-1], 'lx'] * df.loc[df.index[i-1], 'px']
    
    df['dx'] = df['lx'] * df['qx']
    
    df['Lx'] = df['lx'] - (1 - df['ax']) * df['dx']
    
    last_idx = df.index[-1]
    if df.loc[last_idx, 'mx'] > 0:
        df.loc[last_idx, 'Lx'] = df.loc[last_idx, 'lx'] / df.loc[last_idx, 'mx']
    
    df['Tx'] = df['Lx'][::-1].cumsum()[::-1]
    df['ex'] = df['Tx'] / df['lx']
    
    return df

def compute_life_expectancy(mx_future_df):
    e0_results = {}
    
    for year in mx_future_df.columns:
        df_year = pd.DataFrame({
            'Age': mx_future_df.index,
            'mx': mx_future_df[year].values
        })
        
        df_year['ax'] = 0.5
        df_year['qx'] = df_year['mx'] / (1 + 0.5 * df_year['mx'])
        
        lt = compute_life_table(df_year)
        e0_results[year] = lt.iloc[0]['ex']
        
    return pd.Series(e0_results)