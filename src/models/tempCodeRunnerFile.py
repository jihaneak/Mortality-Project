import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm


def split_kt(kt_series, train_end_year):
    """
    Split kt into train and test periods
    """
    kt_train = kt_series[kt_series.index <= train_end_year]
kt_test = kt_series[kt_series.index > train_end_year]

if len(kt_test) == 0:
    raise ValueError(
        f"No test data available. train_end_year={train_end_year} "
        f"but last year in data is {kt_series.index.max()}"
    )


def forecast_kt_arima(kt_train, steps):
    import pmdarima as pm
    import pandas as pd
    
    if steps <= 0:
        raise ValueError("Forecast horizon must be > 0")
    
    model = pm.auto_arima(
        kt_train.values,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )
    
    forecast_values = model.predict(n_periods=steps)
    
    # Build year index
    last_year = kt_train.index[-1]
    forecast_index = range(last_year + 1, last_year + 1 + steps)
    
    forecast_series = pd.Series(forecast_values, index=forecast_index)
    
    return forecast_series, model

def reconstruct_mx(ax, bx, kt_series):
    """
    Reconstruct mx matrix from Lee-Carter parameters
    Returns DataFrame: index=Age, columns=Year
    """
    ages = ax.index
    years = kt_series.index
    
    mx = pd.DataFrame(index=ages, columns=years)
    
    for year in years:
        log_mx = ax + bx * kt_series.loc[year]
        mx[year] = np.exp(log_mx)
        
    return mx

def backtest_life_expectancy(
    ax,
    bx,
    kt_series,
    mx_observed,
    ax_values,
    compute_life_table,
    train_end_year
):
    """
    Full backtesting pipeline
    """
    # Split kt
    kt_train, kt_test = split_kt(kt_series, train_end_year)
    
    # Forecast kt
    steps = len(kt_test)
    kt_forecast, model = forecast_kt_arima(kt_train, steps)
    kt_forecast.index = kt_test.index
    
    # Reconstruct mx
    mx_forecast = reconstruct_mx(ax, bx, kt_forecast)
    
    # Compute life expectancy
    e0_obs = []
    e0_forecast = []
    
    for year in kt_test.index:
        # Observed
        e_obs = life_expectancy_from_mx(
            mx_observed[year],
            ax_values,
            compute_life_table
        )
        
        # Forecast
        e_pred = life_expectancy_from_mx(
            mx_forecast[year],
            ax_values,
            compute_life_table
        )
        
        e0_obs.append(e_obs)
        e0_forecast.append(e_pred)
    
    e0_obs = np.array(e0_obs)
    e0_forecast = np.array(e0_forecast)
    
    # Errors
    errors = e0_forecast - e0_obs
    bias = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    results = {
        "years": kt_test.index,
        "e0_obs": e0_obs,
        "e0_forecast": e0_forecast,
        "bias": bias,
        "rmse": rmse
    }
    
    return results

