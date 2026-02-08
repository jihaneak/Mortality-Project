import numpy as np
from scipy.stats import chi2

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
    df["mx"] = D / E

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
