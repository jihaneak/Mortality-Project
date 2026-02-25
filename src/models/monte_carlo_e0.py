import numpy as np
import pandas as pd
from src.models.life_expectancy import compute_life_table

def monte_carlo_e0(df_mx_future, exposures=None, n_sim=1000, seed=42):
    """
    df_mx_future : DataFrame avec colonnes ['Age','Sex','Year_Future','mx_future','ax']
    exposures : facultatif, sinon on prend 1 pour standardiser
    n_sim : nombre de simulations
    """
    np.random.seed(seed)
    results = []

    for sex in df_mx_future['Sex'].unique():
        df_sex = df_mx_future[df_mx_future['Sex']==sex].copy()
        years = df_sex['Year_Future'].unique()

        for year in years:
            df_year = df_sex[df_sex['Year_Future']==year].copy()
            ages = df_year['Age'].values
            mx = df_year['mx_future'].values
            ax = df_year['ax'].values
            if exposures is None:
                exps = np.ones_like(mx)
            else:
                exps = exposures[df_year.index].values

            e0_sims = []
            for _ in range(n_sim):
                # Simuler les décès
                deaths_sim = np.random.poisson(mx * exps)
                df_sim = pd.DataFrame({
                    'Age': ages,
                    'Sex': sex,
                    'Year_Future': year,
                    'Deaths': deaths_sim,
                    'Exposure': exps,
                    'ax': ax,
                    'mx': deaths_sim / exps
                })
                df_lt = compute_life_table(df_sim)
                e0 = df_lt.loc[df_lt['Age']==0, 'ex'].values[0]
                e0_sims.append(e0)

            e0_lower = np.percentile(e0_sims, 2.5)
            e0_upper = np.percentile(e0_sims, 97.5)
            e0_mean = np.mean(e0_sims)

            results.append({
                'Sex': sex,
                'Year_Future': year,
                'e0_mean': e0_mean,
                'e0_lower': e0_lower,
                'e0_upper': e0_upper
            })

    return pd.DataFrame(results)
