import pandas as pd

def load_data(deaths_path, exposure_path):
    # 1. Load the raw wide data
    deaths = pd.read_csv(deaths_path)
    exposure = pd.read_csv(exposure_path)

    id_cols = ['Year', 'Age']
    sex_cols = ['Female', 'Male', 'Total']

    # 2. Transform Deaths to long format (creates the 'Sex' column)
    deaths_long = deaths.melt(
        id_vars=id_cols, 
        value_vars=sex_cols, 
        var_name='Sex', 
        value_name='Deaths'
    )

    # 3. Transform Exposure to long format (creates the 'Sex' column)
    exposure_long = exposure.melt(
        id_vars=id_cols, 
        value_vars=sex_cols, 
        var_name='Sex', 
        value_name='Exposure'
    )

    # 4. Now that 'Sex' exists in both, we can merge
    df = deaths_long.merge(exposure_long, on=['Year', 'Age', 'Sex'], how='inner')
    
    return df