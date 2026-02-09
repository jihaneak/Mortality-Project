import numpy as np
import pandas as pd

def build_life_table(df, ax=0.5, radix=100_000):
    """
    Build deterministic life table from mx
    Assumes df sorted by Age
    """

    df = df.copy()

    df["qx"] = df["mx"] / (1 + (1 - ax) * df["mx"])
    df.loc[df["qx"] > 1, "qx"] = 1

    df["lx"] = float(radix)
    for i in range(1, len(df)):
        df.loc[i, "lx"] = df.loc[i-1, "lx"] * (1 - df.loc[i-1, "qx"])

    df["dx"] = df["lx"] * df["qx"]
    df["Lx"] = df["lx"] - (1 - ax) * df["dx"]

    df["Tx"] = df["Lx"][::-1].cumsum()[::-1]
    df["ex"] = df["Tx"] / df["lx"]

    return df
