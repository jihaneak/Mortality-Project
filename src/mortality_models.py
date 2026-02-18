import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm 

def poisson_mx(df, alpha=0.05):
    """
    Poisson mortality model:
    Dx ~ Poisson(Ex * mx)

    Returns:
    - mx_hat : MLE
    - mx_ci_lower, mx_ci_upper : exact Poisson CI
    """
    df = df.copy()

    D = df["Deaths"].values
    E = df["Exposure"].values

    # MLE
    with np.errstate(divide='ignore', invalid='ignore'):
        df["mx"] = np.where(E > 0, D / E, 0)
    # Exact Poisson CI for lambda = E * m
    lambda_lower = 0.5 * chi2.ppf(alpha / 2, 2 * D)
    lambda_upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (D + 1))

    df["mx_lower"] = lambda_lower / E
    df["mx_upper"] = lambda_upper / E

    # Asymptotic SE
    df["se_mx"] = np.sqrt(df["mx"] / E)

    return df

def mx_to_qx_delta(df, ax=0.5, alpha=0.05):
    """
    Delta-method transformation from mx to qx
    Assumes asymptotic normality of mx
    """
    df = df.copy()

    mx = df["mx"]
    se_mx = df["se_mx"]

    # Transformation
    df["qx"] = mx / (1 + (1 - ax) * mx)

    # Derivative dq/dm
    dqdm = 1 / (1 + (1 - ax) * mx)**2

    # Delta variance
    var_qx = (dqdm**2) * (se_mx**2)
    df["se_qx"] = np.sqrt(var_qx)

    # Normal CI (delta)
    z = 1.96
    df["qx_lower"] = (df["qx"] - z * df["se_qx"]).clip(0, 1)
    df["qx_upper"] = (df["qx"] + z * df["se_qx"]).clip(0, 1)

    return df

def poisson_mx_exact_ci(df, alpha=0.05):
    """
    Exact Poisson confidence interval for mx using chi-square inversion
    """
    df = df.copy()

    D = df["Deaths"]
    E = df["Exposure"]

    # Avoid zero issues (chi-square with 0 df)
    D_safe = D.copy()
    D_safe[D_safe == 0] = 1e-10

    # Lambda CI
    lambda_lower = 0.5 * chi2.ppf(alpha / 2, 2 * D_safe)
    lambda_upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (D + 1))

    # mx CI
    df["mx_lower_exact"] = lambda_lower / E
    df["mx_upper_exact"] = lambda_upper / E

    return df


def binomial_qx(df, alpha=0.05):
    """
    Binomial mortality model:
    Dx ~ Binomial(Ex, qx)

    Returns qx with Wald confidence interval.
    """

    df = df.copy()

    D = df["Deaths"]
    E = df["Exposure"]

    with np.errstate(divide='ignore', invalid='ignore'):
        # 2. Calculate qx safely
        df["qx_binom"] = np.where(E > 0, D / E, 0)

    # Variance
    var_qx = (df["qx_binom"] * (1 - df["qx_binom"])) / E.replace(0, np.nan)
    df["se_qx_binom"] = np.sqrt(np.where(var_qx > 0, var_qx, 0))

    z = norm.ppf(1 - alpha/2)

    df["qx_binom_lower"] = (df["qx_binom"] - z * df["se_qx_binom"]).clip(0, 1)
    df["qx_binom_upper"] = (df["qx_binom"] + z * df["se_qx_binom"]).clip(0, 1)

    return df

