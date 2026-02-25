import pandas as pd

def read_hmd_csv(deaths_file: str, pop_file: str) -> pd.DataFrame:
    """
    Lit et fusionne les fichiers HMD pour Deaths et Population.
    Renvoie un DataFrame long avec colonnes:
    ['Year', 'Age', 'Sex', 'Deaths', 'Exposure']
    """
    df_d = pd.read_csv(deaths_file)
    df_p = pd.read_csv(pop_file)
    
    df_d_long = df_d.melt(id_vars=['Year', 'Age'], 
                          value_vars=['Female','Male','Total'],
                          var_name='Sex', value_name='Deaths')
    
    df_p_long = df_p.melt(id_vars=['Year','Age'],
                          value_vars=['Female','Male','Total'],
                          var_name='Sex', value_name='Exposure')
    
    df = pd.merge(df_d_long, df_p_long, on=['Year','Age','Sex'])
    
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age','Deaths','Exposure'])
    
    return df