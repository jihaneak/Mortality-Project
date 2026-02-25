import numpy as np

def poisson_mx(df):
    """
    Estimation du taux de mortalité mx et variance sous modèle Poisson
    df doit contenir 'Deaths' et 'Exposure'
    """
    df = df.copy()
    df['mx'] = df['Deaths'] / df['Exposure']
    df['var_mx'] = df['mx'] / df['Exposure']  # Poisson
    df['se_mx'] = np.sqrt(df['var_mx'])
    return df

def delta_qx(df, ax_col='ax'):
    """
    Applique la Delta method pour obtenir l'intervalle de confiance sur qx
    qx = n*mx / (1 + (n - n*ax)*mx) avec n=1
    df doit contenir 'mx' et ax_col
    """
    df = df.copy()
    df['qx'] = df['mx'] / (1 + (1 - df[ax_col]) * df['mx'])
    df['qx'] = df['qx'].clip(upper=1.0)
    
    df['var_qx'] = ((1 / (1 + (1 - df[ax_col]) * df['mx'])**2) ** 2) * df['var_mx']
    df['se_qx'] = np.sqrt(df['var_qx'])
    df['qx_lower'] = (df['qx'] - 1.96*df['se_qx']).clip(lower=0)
    df['qx_upper'] = (df['qx'] + 1.96*df['se_qx']).clip(upper=1)
    
    return df