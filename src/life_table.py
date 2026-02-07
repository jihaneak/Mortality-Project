import numpy as np
import pandas as pd

def ax_constant(df, value=0.5):
    df = df.copy()
    df["ax"] = value
    return df

def mx_to_qx(df):
    df = df.copy()

    df["qx"] = df["mx"] / (1 + (1 - df["ax"]) * df["mx"])

    return df

def qx_delta_method(df):
    df = df.copy()

    derivative = 1 / (1 + (1 - df["ax"]) * df["mx"])**2
    df["se_qx"] = derivative * df["se_mx"]

    return df

def build_life_table(df, radix=100000):
    df = df.copy()

    lx = [radix]
    dx = []
    Lx = []

    for q in df["qx"]:
        d = lx[-1] * q
        dx.append(d)
        lx.append(lx[-1] - d)
        Lx.append(lx[-2] - 0.5 * d)

    df["lx"] = lx[:-1]
    df["dx"] = dx
    df["Lx"] = Lx

    df["Tx"] = np.flip(np.cumsum(np.flip(df["Lx"])))
    df["ex"] = df["Tx"] / df["lx"]

    return df
