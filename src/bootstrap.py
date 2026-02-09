import numpy as np
import pandas as pd
from .life_table import build_life_table

def simulate_deaths(df):
    """
    Simulate deaths under Poisson model
    """
    return np.random.poisson(df["Exposure"] * df["mx"])


def bootstrap_life_expectancy(df, B=1000, ax=0.5, radix=100_000):
    """
    Bootstrap distribution of e0
    """

    e0_samples = []

    for b in range(B):
        sim_df = df.copy()

        sim_df["Deaths_sim"] = simulate_deaths(sim_df)
        sim_df["mx"] = sim_df["Deaths_sim"] / sim_df["Exposure"]

        lt = build_life_table(sim_df, ax=ax, radix=radix)

        e0_samples.append(lt.loc[lt["Age"] == 0, "ex"].values[0])

    return np.array(e0_samples)
