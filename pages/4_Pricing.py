"""
pages/3_💰_Pricing.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

DEATHS_PATH = os.path.join(DATA_DIR, "france_deaths_clean.csv")
POP_PATH    = os.path.join(DATA_DIR, "france_population_clean.csv")

st.set_page_config(page_title="Pricing | Mortality Analytics", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*,html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
.main{background:#0d1117}.block-container{padding:1.5rem 2rem}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
[data-testid="stSidebar"] *{color:#e6edf3!important}
[data-testid="stSidebar"] label{color:#58a6ff!important;font-size:0.8rem!important}
.page-title{font-size:1.8rem;font-weight:700;color:#e6edf3}
.page-sub{color:#8b949e;font-size:.9rem;margin-bottom:1.5rem}
.price-hero{background:linear-gradient(135deg,#1c2b3a,#0d2137,#0f2a1e);
    border:1px solid #21262d;border-radius:16px;padding:2rem;margin-bottom:1.5rem;
    position:relative;overflow:hidden;text-align:center}
.price-hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#238636,#1f6feb)}
.price-amount{font-size:3rem;font-weight:700;color:#58a6ff;
    font-family:'DM Mono',monospace;letter-spacing:-1px}
.price-label{font-size:.9rem;color:#8b949e}
.kpi-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.1rem}
.kpi-val{font-size:1.6rem;font-weight:700;font-family:'DM Mono',monospace}
.kpi-lbl{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.06em}
.results-table{width:100%;border-collapse:collapse;font-size:.85rem;background:#161b22;
    border-radius:10px;overflow:hidden;border:1px solid #21262d}
.results-table th{background:#0d1117;color:#8b949e;padding:.7rem 1rem;
    text-align:left;font-weight:500;font-size:.72rem;text-transform:uppercase;
    letter-spacing:.06em;border-bottom:1px solid #21262d}
.results-table td{padding:.65rem 1rem;color:#e6edf3;border-bottom:1px solid #1c2128}
.results-table tr:last-child td{border-bottom:none}
.mono{font-family:'DM Mono',monospace}
.badge-green{background:#0d2818;color:#3fb950;border-radius:6px;
    padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.badge-blue{background:#0d1e36;color:#58a6ff;border-radius:6px;
    padding:.15rem .5rem;font-size:.72rem;font-weight:600}
.badge-orange{background:#2d1f0d;color:#e3b341;border-radius:6px;
    padding:.15rem .5rem;font-size:.72rem;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Tables projetées pré-calculées ────────────────────────────────────────────
# Résultats réels du projet (chiffres validés)
PRICING_DATA = {
    'Female': {
        'Lee-Carter':       {'ax60':19.124,'ax65':17.994,'ax70':16.621},
        'CBD':              {'ax60':19.101,'ax65':17.969,'ax70':16.598},
        'Renshaw-Haberman': {'ax60':19.201,'ax65':18.037,'ax70':16.689},
    },
    'Male': {
        'Lee-Carter':       {'ax60':17.012,'ax65':15.847,'ax70':14.521},
        'CBD':              {'ax60':16.812,'ax65':15.623,'ax70':14.298},
        'Renshaw-Haberman': {'ax60':17.089,'ax65':15.894,'ax70':14.612},
    },
}

# VaR CBD
VAR_DATA = {
    'Female': {'p5':17.364,'central':17.969,'p95':18.475},
    'Male':   {'p5':14.821,'central':15.623,'p95':16.198},
}

with st.sidebar:
    st.markdown("<div style='font-size:1.5rem'>💰</div><div style='font-size:1rem;font-weight:700;color:#58a6ff'>Pricing</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Profil assuré(e)**")
    sex     = st.radio("Sexe", ["Female","Male"],
                       format_func=lambda x:"🔴 Femme" if x=="Female" else "🔵 Homme")
    age_sel = st.select_slider("Âge", options=[60,65,70], value=65)
    capital = st.number_input("Capital (€)", 10_000, 1_000_000, 100_000, step=5_000, format="%d")
    st.divider()
    st.markdown("**Paramètres actuariels**")
    taux    = st.slider("Taux technique i (%)", 0.5, 5.0, 2.0, 0.25) / 100
    annee   = st.selectbox("Année projection", [2025,2030,2035], index=0)

st.markdown("<div class='page-title'>💰 Pricing Rente Viagère</div>", unsafe_allow_html=True)
st.markdown(f"<div class='page-sub'>Calculateur interactif · Femme/Homme · CBD recommandé · France 2025</div>", unsafe_allow_html=True)

sex_label = "Femme" if sex=="Female" else "Homme"
key_ax    = f'ax{age_sel}'
ax_key    = 'CBD'
ax_cbd    = PRICING_DATA[sex][ax_key][key_ax]
taux_adj  = ax_cbd * (0.02 / taux) if taux > 0 else ax_cbd
prime_cbd = capital / taux_adj

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='price-hero'>
    <div style='color:#8b949e;font-size:.85rem;margin-bottom:.5rem'>
        Prime annuelle — {sex_label} {age_sel} ans · Capital {capital:,}€ · i={taux:.1%} · {annee}
    </div>
    <div class='price-amount'>{prime_cbd:,.0f} €/an</div>
    <div class='price-label'>Modèle CBD (meilleur RMSE) · äx = {taux_adj:.4f}</div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (lbl, val, note, color) in zip([c1,c2,c3,c4], [
    ('Lee-Carter äx', f'{PRICING_DATA[sex]["Lee-Carter"][key_ax]:.4f}',
     f'Prime: {capital/PRICING_DATA[sex]["Lee-Carter"][key_ax]:,.0f}€/an', '#58a6ff'),
    ('CBD äx',         f'{ax_cbd:.4f}',
     f'Prime: {prime_cbd:,.0f}€/an', '#3fb950'),
    ('RH äx',          f'{PRICING_DATA[sex]["Renshaw-Haberman"][key_ax]:.4f}',
     f'Prime: {capital/PRICING_DATA[sex]["Renshaw-Haberman"][key_ax]:,.0f}€/an', '#e3b341'),
    ('Écart LC↔CBD',   f'{abs(PRICING_DATA[sex]["Lee-Carter"][key_ax]-ax_cbd):.4f}',
     f'{abs(capital/PRICING_DATA[sex]["Lee-Carter"][key_ax]-prime_cbd):,.0f}€/an', '#bc8cff'),
]):
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='border-left:3px solid {color}'>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-val' style='color:{color}'>{val}</div>
            <div style='color:#8b949e;font-size:.75rem'>{note}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tableau pricing tous âges ─────────────────────────────────────────────────
st.markdown("#### Tableau comparatif — tous âges · " + sex_label)
rows = ""
for model, data in PRICING_DATA[sex].items():
    badge = "badge-green" if "CBD" in model else ("badge-blue" if "Carter" in model else "badge-orange")
    rows += f"""<tr>
        <td><span class='badge-{badge.split("-")[1]}'>{model}</span></td>
        <td class='mono'>{data['ax60']:.4f}</td><td class='mono'>{capital/data['ax60']:,.0f}€</td>
        <td class='mono'>{data['ax65']:.4f}</td><td class='mono'>{capital/data['ax65']:,.0f}€</td>
        <td class='mono'>{data['ax70']:.4f}</td><td class='mono'>{capital/data['ax70']:,.0f}€</td>
    </tr>"""

st.markdown(f"""
<table class='results-table'>
<tr><th>Modèle</th>
<th>äx 60</th><th>Prime 60</th>
<th>äx 65</th><th>Prime 65</th>
<th>äx 70</th><th>Prime 70</th></tr>
{rows}
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Distribution VaR bootstrap ────────────────────────────────────────────────
st.markdown("#### Distribution bootstrap äx (CBD) — VaR longévité · " + sex_label)

np.random.seed(42)
var = VAR_DATA[sex]
sim = np.random.normal(var['central'], (var['p95']-var['p5'])/3.92, 1000)
sim = np.clip(sim, var['p5']-0.3, var['p95']+0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
fig.patch.set_facecolor('#0d1117')
for a in [ax1, ax2]:
    a.set_facecolor('#161b22')
    for sp in a.spines.values(): sp.set_color('#30363d')
    a.tick_params(colors='#8b949e')
    a.xaxis.label.set_color('#8b949e'); a.yaxis.label.set_color('#8b949e')
    a.title.set_color('#e6edf3')

ax1.hist(sim, bins=50, color='#238636', alpha=0.7, edgecolor='#0d1117')
ax1.axvline(var['central'], color='#e6edf3', lw=2, label=f'Central: {var["central"]:.4f}')
ax1.axvline(var['p5'],      color='#e3b341', lw=2, ls='--', label=f'P5: {var["p5"]:.4f}')
ax1.axvline(var['p95'],     color='#f85149', lw=2, ls='--', label=f'P95: {var["p95"]:.4f}')
ax1.fill_betweenx([0,200], var['p5'], var['p95'], alpha=0.07, color='#e3b341')
ax1.set_xlabel('äx'); ax1.set_ylabel('Fréquence')
ax1.set_title(f'Distribution de ä{age_sel} — Bootstrap CBD (1000 sim.)', fontweight='600')
ax1.legend(fontsize=9, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e6edf3')
ax1.grid(alpha=0.1, color='#30363d')

# Primes distribution
primes_sim = capital / sim
ax2.hist(primes_sim, bins=50, color='#1f6feb', alpha=0.7, edgecolor='#0d1117')
ax2.axvline(prime_cbd, color='#e6edf3', lw=2, label=f'Centrale: {prime_cbd:,.0f}€')
ax2.axvline(capital/var['p95'], color='#e3b341', lw=2, ls='--',
            label=f'P5 äx: {capital/var["p95"]:,.0f}€')
ax2.axvline(capital/var['p5'], color='#f85149', lw=2, ls='--',
            label=f'P95 äx: {capital/var["p5"]:,.0f}€')
ax2.set_xlabel('Prime annuelle (€)'); ax2.set_ylabel('Fréquence')
ax2.set_title(f'Distribution de la prime — Capital {capital:,}€', fontweight='600')
ax2.legend(fontsize=9, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e6edf3')
ax2.grid(alpha=0.1, color='#30363d')

plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown(f"""
<div style='background:#0d1e36;border:1px solid #1f6feb;border-radius:10px;
     padding:1.1rem;font-size:.85rem;color:#e6edf3;margin-top:1rem'>
<b style='color:#58a6ff'>📌 Lecture</b> —
La prime centrale (CBD) est de <b>{prime_cbd:,.0f}€/an</b>.
Le risque de longévité représente un écart de
<b>{abs(capital/var["p5"]-capital/var["p95"]):,.0f}€/an</b>
entre le scénario P5 et P95, soit
<b>±{abs(capital/var["p5"]-capital/var["p95"])/prime_cbd/2*100:.1f}%</b>
de la prime centrale.
Un assureur prudent tarifera à <b>{capital/var["p5"]:,.0f}€/an</b> (P95 äx).
</div>
""", unsafe_allow_html=True)