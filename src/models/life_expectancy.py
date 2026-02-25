import pandas as pd
import numpy as np

def compute_life_table(df):
    """
    Calcule une table de mortalité complète pour une seule année.
    Version corrigée avec fermeture actuarielle.
    """
    df = df.reset_index(drop=True).sort_values('Age').copy()
    
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