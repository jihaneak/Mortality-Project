import numpy as np
import pandas as pd

from src.models.life_expectancy import compute_life_table


def monte_carlo_e0(df_mx_future, exposures=None, n_sim=1000, seed=42):
    """
    Simule la distribution de e0 par Monte Carlo (modèle Poisson sur les décès).

    Paramètres
    ----------
    df_mx_future : DataFrame avec colonnes Sex, Year_Future, Age, mx_future, ax
    exposures    : Series avec les expositions (index aligné sur df_mx_future) ;
                   si None, on suppose des expositions de 1 (usage normatif)
    n_sim        : nombre de simulations
    seed         : graine aléatoire pour reproductibilité

    Retourne
    --------
    DataFrame avec colonnes : Sex, Year_Future, e0_mean, e0_lower, e0_upper
    """
    np.random.seed(seed)
    results = []

    for sex in df_mx_future['Sex'].unique():
        df_sex = df_mx_future[df_mx_future['Sex'] == sex].copy()

        for year in df_sex['Year_Future'].unique():
            df_year = df_sex[df_sex['Year_Future'] == year].copy()

            ages = df_year['Age'].values
            mx   = df_year['mx_future'].values
            ax   = df_year['ax'].values

            exps = (
                exposures[df_year.index].values
                if exposures is not None
                else np.ones_like(mx)
            )

            e0_sims = []
            for _ in range(n_sim):
                # Simuler les décès sous modèle Poisson
                deaths_sim = np.random.poisson(mx * exps)
                mx_sim     = (deaths_sim / exps).clip(min=1e-10)

                df_sim = pd.DataFrame({
                    'Age': ages,
                    'mx':  mx_sim,
                    'ax':  ax,
                })
                lt  = compute_life_table(df_sim)
                e0_sims.append(lt.iloc[0]['ex'])

            e0_sims = np.array(e0_sims)
            results.append({
                'Sex':         sex,
                'Year_Future': year,
                'e0_mean':     float(np.mean(e0_sims)),
                'e0_lower':    float(np.percentile(e0_sims, 2.5)),
                'e0_upper':    float(np.percentile(e0_sims, 97.5)),
            })

    return pd.DataFrame(results)