import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.lee_carter import fit_lee_carter, reconstruct_mx, forecast_kt
from src.models.life_expectancy import compute_life_table

st.set_page_config(page_title="DeepActuary France", layout="wide")
st.title("🇫🇷 DeepActuary : Analyse de la Mortalité")

@st.cache_data
def load_and_merge_data():
    # 1. Chargement des fichiers séparés
    df_deaths = pd.read_csv("data/france_deaths_clean.csv")
    df_pop = pd.read_csv("data/france_population_clean.csv")
    
    # 2. Transformation si tes fichiers ont des colonnes 'Male' / 'Female'
    # On utilise melt pour passer au format "long" compatible Lee-Carter
    deaths_long = df_deaths.melt(id_vars=['Year', 'Age'], value_vars=['Male', 'Female'], 
                                 var_name='Sex', value_name='Deaths')
    pop_long = df_pop.melt(id_vars=['Year', 'Age'], value_vars=['Male', 'Female'], 
                           var_name='Sex', value_name='Population')
    
    # 3. Fusion et calcul du taux mx
    df_merged = pd.merge(deaths_long, pop_long, on=['Year', 'Age', 'Sex'])
    df_merged['mx'] = df_merged['Deaths'] / df_merged['Population']
    
    return df_merged

try:
    df_full = load_and_merge_data()
    
    # --- Interface Sidebar ---
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
    horizon = st.sidebar.slider("Horizon de projection", 10, 50, 30)
    
    # Filtrage pour la calibration
    df_sub = df_full[df_full['Sex'] == gender].copy()
    
    # --- Calibration & Projection ---
    ax, bx, kt = fit_lee_carter(df_sub)
    drift = (kt[-1] - kt[0]) / (len(kt) - 1)
    
    steps = np.arange(1, horizon + 1)
    kt_proj = kt[-1] + (drift * steps)

    # --- Graphiques ---
    st.subheader(fr"Projection du paramètre temporel $\kappa_t$ ({gender})")
    fig, ax_kt = plt.subplots(figsize=(10, 4))
    ax_kt.plot(kt, color='black', label="Historique")
    ax_kt.plot(range(len(kt), len(kt)+horizon), kt_proj, '--', color='blue', label="Projection")
    ax_kt.set_title(fr"Trajectoire de $\kappa_t$ (Lee-Carter)")
    ax_kt.legend()
    st.pyplot(fig)

    # --- Résultat Métrique ---
    mx_f = reconstruct_mx(ax, bx, [kt_proj[-1]])
    df_lt = mx_f.iloc[:, 0].to_frame(name='mx').reset_index().rename(columns={'index':'Age'})
    df_lt['ax'], df_lt['qx'] = 0.5, df_lt['mx']/(1+0.5*df_lt['mx'])
    lt = compute_life_table(df_lt)
    st.metric(f"Espérance de vie estimée en {2020+horizon}", f"{lt.iloc[0]['ex']:.2f} ans")

except Exception as e:
    st.error(f"Erreur lors du traitement : {e}")