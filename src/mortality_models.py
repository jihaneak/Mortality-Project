import numpy as np

def poisson_mx(df):
    """
    Estimate mortality intensity under Poisson model
    """
    df = df.copy()

    df["mx"] = df["Deaths"] / df["Exposure"]
    df["var_mx"] = df["mx"] / df["Exposure"]
    df["se_mx"] = np.sqrt(df["var_mx"])

    return df
