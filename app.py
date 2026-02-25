import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.lee_carter import fit_lee_carter, reconstruct_mx
from src.models.life_expectancy import compute_life_table

st.set_page_config(page_title="Mortality in France", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] { background-color: #1e2130; border: 1px solid #31333f; border-radius: 10px; padding: 15px; }
    h1, h2, h3 { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df_d = pd.read_csv("data/france_deaths_clean.csv")
        df_p = pd.read_csv("data/france_population_clean.csv")
        d_l = df_d.melt(id_vars=['Year', 'Age'], value_vars=['Male', 'Female'], var_name='Sex', value_name='Deaths')
        p_l = df_p.melt(id_vars=['Year', 'Age'], value_vars=['Male', 'Female'], var_name='Sex', value_name='Population')
        df = pd.merge(d_l, p_l, on=['Year', 'Age', 'Sex'])
        df['mx'] = df['Deaths'] / df['Population']
        return df
    except Exception as e:
        st.error(f"Fichiers manquants : {e}")
        return None

with st.sidebar:
    st.title("🛡️ Settings")
    gender = st.selectbox("Target Gender", ["Female", "Male"])
    horizon = st.slider("Forecast Horizon", 10, 50, 30)
    st.info("Modèle : Lee-Carter Stochastic Projection")

df_full = load_data()

if df_full is not None:
    df_sub = df_full[df_full['Sex'] == gender].copy()
    
    ax_param, bx_param, kt_param = fit_lee_carter(df_sub)
    drift = (kt_param[-1] - kt_param[0]) / (len(kt_param) - 1)
    kt_proj = kt_param[-1] + (drift * np.arange(1, horizon + 1))
    
    mx_f = reconstruct_mx(ax_param, bx_param, [kt_proj[-1]])
    df_lt_in = mx_f.iloc[:, 0].to_frame(name='mx').reset_index().rename(columns={'index':'Age'})
    df_lt_in['ax'], df_lt_in['qx'] = 0.5, df_lt_in['mx']/(1+0.5*df_lt_in['mx'])
    lt_final = compute_life_table(df_lt_in)

    st.header(f"Mortality Analytics Dashboard : {gender}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Projected e0", f"{lt_final.iloc[0]['ex']:.2f} yrs")
    k2.metric("Drift (κ)", f"{drift:.4f}")
    k3.metric("Data Max Year", f"{df_sub['Year'].max()}")

    st.markdown("---")

    tab_trend, tab_bio, tab_heat = st.tabs(["📉 Projections", "🧬 Paramètres Bio", "🔥 Lexis Surface"])

    with tab_trend:
        st.subheader("Long-term Mortality Trend (κt)")
        fig_kt, ax_kt = plt.subplots(figsize=(15, 5), facecolor='#0e1117')
        ax_kt.set_facecolor('#0e1117')
        ax_kt.plot(kt_param, color='#00d4ff', label="Historical")
        ax_kt.plot(range(len(kt_param), len(kt_param)+horizon), kt_proj, '--', color='#ff9f1c', label="Forecast")
        ax_kt.tick_params(colors='white')
        ax_kt.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig_kt)

        st.subheader("Poisson Uncertainty (Current Year)")
        df_last = df_sub[df_sub['Year'] == df_sub['Year'].max()].set_index('Age')
        std_err = np.sqrt(df_last['mx'] / df_last['Population']) 
        
        fig_p, ax_p = plt.subplots(figsize=(15, 5), facecolor='#0e1117')
        ax_p.set_facecolor('#0e1117')
        ax_p.plot(df_last.index, df_last['mx'], color='#ff4b4b', label="Observed mx")
        ax_p.fill_between(df_last.index, df_last['mx']-1.96*std_err, df_last['mx']+1.96*std_err, color='#ff4b4b', alpha=0.2)
        ax_p.set_yscale('log')
        ax_p.tick_params(colors='white')
        st.pyplot(fig_p)

    with tab_bio:
        st.subheader("Lee-Carter Biological Components")
        c_a, c_b = st.columns(2)
        with c_a:
            fig_a, ax_a = plt.subplots(facecolor='#0e1117')
            ax_a.set_facecolor('#0e1117')
            ax_a.plot(ax_param, color='#00d4ff')
            ax_a.set_title("Average Mortality Profile (ax)", color='white')
            ax_a.tick_params(colors='white')
            st.pyplot(fig_a)
        with c_b:
            fig_b, ax_b = plt.subplots(facecolor='#0e1117')
            ax_b.set_facecolor('#0e1117')
            ax_b.plot(bx_param, color='#ff9f1c')
            ax_b.set_title("Age Sensitivity to Progress (bx)", color='white')
            ax_b.tick_params(colors='white')
            st.pyplot(fig_b)

    with tab_heat:
        st.subheader("Lexis Surface (Log-Mortality)")
        try:
            heat_data = df_sub.pivot_table(index='Age', columns='Year', values='mx')
            log_mx = np.log10(heat_data + 1e-10) # Log10 pour le contraste
            fig_h, ax_h = plt.subplots(figsize=(15, 8), facecolor='#0e1117')
            sns.heatmap(log_mx, cmap="magma", ax=ax_h, vmin=-5, vmax=-0.5, cbar_kws={'label': 'Log10 Mortality'})
            ax_h.tick_params(colors='white')
            ax_h.set_xlabel("Year", color='white')
            ax_h.set_ylabel("Age", color='white')
            st.pyplot(fig_h)
        except Exception as e:
            st.error(f"Erreur Heatmap : {e}")

    with st.expander("📋 View Forecasted Life Table"):
        st.dataframe(lt_final.style.background_gradient(cmap='Blues'), use_container_width=True)

else:
    st.warning("Vérifiez la présence des fichiers dans le dossier data/")