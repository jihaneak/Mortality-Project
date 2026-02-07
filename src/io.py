import pandas as pd

def load_data(deaths_path, exposure_path):
    deaths = pd.read_csv(deaths_path)
    exposure = pd.read_csv(exposure_path)

    df = deaths.merge(
        exposure,
        on=["Year", "Age", "Sex"],
        how="inner"
    )
    return df
