import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

DEATHS_PATH = os.path.join(DATA_DIR, "france_deaths_clean.csv")
POP_PATH    = os.path.join(DATA_DIR, "france_population_clean.csv")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèles | Mortality Analytics",
                   page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] label { color: #58a6ff !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

DEATHS_PATH = r"C:\Users\PC-HP\Desktop\insea\Mortality Project\data\france_deaths_clean.csv"
POP_PATH    = r"C:\Users\PC-HP\Desktop\insea\Mortality Project\data\france_population_clean.csv"
COLORS_SEX  = {'Female': '#f85149', 'Male': '#58a6ff'}

def dark_ax(a):
    a.set_facecolor('#161b22')
    for sp in a.spines.values(): sp.set_color('#30363d')
    a.tick_params(colors='#8b949e')
    a.xaxis.label.set_color('#8b949e')
    a.yaxis.label.set_color('#8b949e')
    a.title.set_color('#e6edf3')

TH = ("style='background:#0d1117;color:#8b949e;padding:.7rem 1rem;"
      "text-align:left;font-size:.72rem;text-transform:uppercase;"
      "letter-spacing:.06em;border-bottom:1px solid #21262d'")
TD = "style='padding:.65rem 1rem;color:#e6edf3;border-bottom:1px solid #1c2128'"
TABLE = ("style='width:100%;border-collapse:collapse;font-size:.85rem;"
         "background:#161b22;border-radius:10px;overflow:hidden;"
         "border:1px solid #21262d'")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;border-bottom:1px solid #30363d;margin-bottom:1rem'>
        <div style='font-size:.95rem;font-weight:600;color:#e6edf3'>Mortality Analytics</div>
        <div style='font-size:.72rem;color:#8b949e;margin-top:.2rem'>
            INSEA — Statistiques & Démographie</div>
    </div>""", unsafe_allow_html=True)
    sex = st.radio("Sexe", ["Female", "Male"],
                   format_func=lambda x: "Femmes" if x == "Female" else "Hommes")

sex_label = 'Femmes' if sex == 'Female' else 'Hommes'

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e6edf3;font-weight:700'>🔬 Comparaison des modèles</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:.9rem;margin-bottom:1.5rem'>"
            "Backtest rolling one-step-ahead 2001–2020 · Résidus · Paramètres CBD</p>",
            unsafe_allow_html=True)

# ── Tableau résultats ─────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:#e6edf3'>Résultats backtesting — {sex_label}</h4>",
            unsafe_allow_html=True)

metrics = {
    'Female': [
        ('Lee-Carter (LC1)', '0.463', '+0.148', '82.6%', '#58a6ff', '#e3b341', False, False),
        ('LC2',              '0.601', '+0.374', '78.3%', '#bc8cff', '#e3b341', False, False),
        ('CBD ★',            '0.188', '+0.084', '91.3%', '#3fb950', '#3fb950', True,  False),
        ('Renshaw-Haberman', '0.384', '+0.052', '87.0%', '#e3b341', '#3fb950', False, True),
        ('LC Bayésien (KF)', '0.482', '+0.139', '95.7%', '#bc8cff', '#3fb950', False, False),
    ],
    'Male': [
        ('Lee-Carter (LC1)', '0.636', '−0.552', '47.8%', '#58a6ff', '#f85149', False, False),
        ('LC2',              '0.556', '−0.441', '69.6%', '#bc8cff', '#e3b341', False, False),
        ('CBD ★',            '0.228', '−0.087', '95.7%', '#3fb950', '#3fb950', True,  False),
        ('Renshaw-Haberman', '0.562', '−0.471', '60.9%', '#e3b341', '#f85149', False, False),
        ('LC Bayésien (KF)', '0.692', '−0.624', '65.2%', '#bc8cff', '#e3b341', False, False),
    ],
}

rows = ""
for name, rmse, bias, cov, col_name, col_cov, best_r, best_b in metrics[sex]:
    rmse_style = f"color:#3fb950;font-weight:600;font-family:'DM Mono',monospace" if best_r else "font-family:'DM Mono',monospace;color:#e6edf3"
    bias_style = f"color:#3fb950;font-weight:600;font-family:'DM Mono',monospace" if best_b else "font-family:'DM Mono',monospace;color:#e6edf3"
    name_badge = (f"<span style='display:inline-block;background:#0d2818;color:{col_name};"
                  f"border-radius:6px;padding:.15rem .5rem;font-size:.8rem;font-weight:600'>{name}</span>")
    cov_badge  = (f"<span style='display:inline-block;background:#161b22;color:{col_cov};"
                  f"border-radius:6px;padding:.15rem .5rem;font-size:.8rem;font-weight:600;"
                  f"border:1px solid {col_cov}33'>{cov}</span>")
    rows += (f"<tr>"
             f"<td {TD}>{name_badge}</td>"
             f"<td style='padding:.65rem 1rem;border-bottom:1px solid #1c2128;{rmse_style}'>{rmse}</td>"
             f"<td style='padding:.65rem 1rem;border-bottom:1px solid #1c2128;{bias_style}'>{bias}</td>"
             f"<td {TD}>{cov_badge}</td>"
             f"</tr>")

st.markdown(
    f"<table {TABLE}>"
    f"<tr><th {TH}>Modèle</th><th {TH}>RMSE (ans)</th>"
    f"<th {TH}>Biais (ans)</th><th {TH}>Coverage IC 95%</th></tr>"
    f"{rows}</table>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Paramètres CBD ────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:#e6edf3'>Paramètres CBD κ1/κ2 — {sex_label}</h4>",
            unsafe_allow_html=True)

cbd = {
    'Female': {'trend_k1': -1.019, 'trend_k2': -0.0029, 'k2_1950': 0.1055, 'res': 0.1342},
    'Male':   {'trend_k1': -0.666, 'trend_k2': -0.0008, 'k2_1950': 0.0905, 'res': 0.0832},
}
d = cbd[sex]
c1, c2, c3, c4 = st.columns(4)
for col, (lbl, val, color) in zip([c1, c2, c3, c4], [
    ('κ1 tendance 1950→2000', f'{d["trend_k1"]:+.3f}', '#f85149'),
    ('κ2 tendance 1950→2000', f'{d["trend_k2"]:+.4f}', '#e3b341'),
    ('κ2 en 1950',            f'{d["k2_1950"]:.4f}',   '#58a6ff'),
    ('Résidu std (logit)',     f'{d["res"]:.4f}',        '#3fb950'),
]):
    with col:
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #21262d;border-radius:10px;
             padding:1.1rem;border-left:3px solid {color}'>
            <div style='font-size:.72rem;color:#8b949e;text-transform:uppercase;
                        letter-spacing:.06em'>{lbl}</div>
            <div style='font-size:1.5rem;font-weight:700;font-family:DM Mono,monospace;
                        color:{color}'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Résidus LC ────────────────────────────────────────────────────────────────
st.markdown(f"<h4 style='color:#e6edf3'>Carte des résidus Lee-Carter — {sex_label}</h4>",
            unsafe_allow_html=True)

try:
    from src.models.lee_carter import fit_lee_carter

    @st.cache_data
    def load_train(sex_key):
        df_d = pd.read_csv(DEATHS_PATH)
        df_p = pd.read_csv(POP_PATH)
        d_l  = df_d.melt(id_vars=['Year','Age'], value_vars=['Male','Female'],
                         var_name='Sex', value_name='Deaths')
        p_l  = df_p.melt(id_vars=['Year','Age'], value_vars=['Male','Female'],
                         var_name='Sex', value_name='Population')
        df   = pd.merge(d_l, p_l, on=['Year','Age','Sex'])
        df['mx'] = df['Deaths'] / df['Population']
        df = df[(df['mx'] > 0) & (df['Age'] <= 90) & (df['Year'] >= 1950)].copy()
        return df[(df['Year'] <= 2000) & (df['Sex'] == sex_key)].copy()

    df_tr = load_train(sex)
    ax_lc, bx_lc, kt_lc = fit_lee_carter(df_tr)
    pivot     = df_tr.pivot(index='Age', columns='Year', values='mx')
    log_obs   = np.log(pivot.clip(lower=1e-10).values)
    log_fit   = np.column_stack([ax_lc.values + bx_lc.values * kt_lc[y] for y in kt_lc.index])
    residuals = log_fit - log_obs
    resid_std = float(np.std(residuals.ravel()))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    im = ax.imshow(residuals, aspect='auto', cmap='RdBu', vmin=-0.3, vmax=0.3,
                   extent=[kt_lc.index[0], kt_lc.index[-1], 90, 0])
    cb = plt.colorbar(im, ax=ax, label='Résidu log(mx)')
    cb.ax.tick_params(labelcolor='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e')
    ax.set_xlabel('Année', color='#8b949e')
    ax.set_ylabel('Âge', color='#8b949e')
    ax.set_title(f'{sex_label} — Résidu std = {resid_std:.4f}',
                 fontweight='600', color='#e6edf3')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # kt plot
    st.markdown(f"<h4 style='color:#e6edf3'>Indice temporel kt — {sex_label}</h4>",
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    fig2.patch.set_facecolor('#0d1117')
    dark_ax(ax2)
    kt_lc.plot(ax=ax2, color=COLORS_SEX[sex], lw=2)
    ax2.set_xlabel('Année')
    ax2.set_ylabel('kt')
    ax2.set_title(f'kt — {sex_label} | Drift = {np.mean(np.diff(kt_lc.values)):.3f}/an',
                  fontweight='600', color='#e6edf3')
    ax2.grid(alpha=0.1, color='#30363d')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

except Exception as e:
    resid_std_display = '0.0975' if sex == 'Female' else '0.0980'
    st.markdown(f"""
    <div style='background:#0d1e36;border:1px solid #1f6feb;border-radius:10px;
         padding:1.2rem;color:#8b949e;font-size:.85rem'>
    ℹ️ Résidus dynamiques indisponibles : {e}<br><br>
    Résidu std {sex_label} = <b style='color:#e6edf3'>{resid_std_display}</b>
    </div>""", unsafe_allow_html=True)

# ── Interprétation ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0d2818;border:1px solid #238636;border-radius:10px;
     padding:1.1rem;font-size:.85rem;color:#e6edf3;margin-top:1rem'>
<b style='color:#3fb950'>💡 Lecture de la heatmap</b><br><br>
🔴 <b>Diagonale rouge (âges 40–70)</b> — Effet de cohorte :
générations nées 1910–1930 (WWI, grippe espagnole 1918).<br>
🔵 <b>Bloc bleu vieux âges post-1985</b> — Révolution cardiovasculaire
sous-estimée (statines, pontages).<br>
🔴 <b>Bloc rouge hommes 15–35 ans</b> — Épidémie SIDA + accidents de la route.
</div>
""", unsafe_allow_html=True)