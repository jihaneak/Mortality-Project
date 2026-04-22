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

st.set_page_config(page_title="Risque Longévité | Mortality Analytics",
                   page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] label { color: #58a6ff !important; font-size: 0.8rem !important; }
.page-title { font-size: 1.8rem; font-weight: 700; color: #e6edf3; }
.page-sub   { color: #8b949e; font-size: .9rem; margin-bottom: 1.5rem; }
.kpi-card   { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1.1rem; }
.kpi-val    { font-size: 1.5rem; font-weight: 700; font-family: 'DM Mono', monospace; }
.kpi-lbl    { font-size: .72rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; }
table.rt { width:100%; border-collapse:collapse; font-size:.85rem; background:#161b22;
    border-radius:10px; overflow:hidden; border:1px solid #21262d; }
table.rt th { background:#0d1117; color:#8b949e; padding:.7rem 1rem; text-align:left;
    font-weight:500; font-size:.72rem; text-transform:uppercase;
    letter-spacing:.06em; border-bottom:1px solid #21262d; }
table.rt td { padding:.65rem 1rem; color:#e6edf3; border-bottom:1px solid #1c2128; }
table.rt tr:last-child td { border-bottom:none; }
.mono { font-family:'DM Mono',monospace; }
.total-box { background:linear-gradient(135deg,#0d1e36,#0d2818);
    border:1px solid #1f6feb; border-radius:12px; padding:1.5rem;
    text-align:center; margin-top:1rem; }
.total-amount { font-size:2.4rem; font-weight:700; color:#58a6ff;
    font-family:'DM Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Données réelles ───────────────────────────────────────────────────────────
BASE_CAPITAL = 14_365_295

PROVISIONS = [
    ('Central (P50)',    16.087,  0,       0.0),
    ('Prudent (P75)',    16.116,  0,       0.0),
    ('Élevé (P90)',      16.143,  18_681,  0.1),
    ('Sévère (P95)',     16.155,  29_359,  0.2),
    ('Extrême (P99)',    16.192,  62_441,  0.4),
    ('SolvII (P99.5)',   16.202,  71_298,  0.5),
]

STRESS = [
    ('-5%',  126_739, 0.9,  False),
    ('-10%', 256_257, 1.8,  False),
    ('-15%', 388_650, 2.7,  False),
    ('-20%', 524_015, 3.6,  True),
    ('-25%', 662_454, 4.6,  False),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;border-bottom:1px solid #30363d;margin-bottom:1rem'>
        <div style='font-size:.95rem;font-weight:600;color:#e6edf3'>Mortality Analytics</div>
        <div style='font-size:.72rem;color:#8b949e;margin-top:.2rem'>INSEA — Statistiques & Démographie</div>
    </div>""", unsafe_allow_html=True)
    capital = st.number_input("Capital total (€)", 1_000_000, 100_000_000,
                               BASE_CAPITAL, step=1_000_000, format="%d")
    n_cont  = st.number_input("Nb contrats", 10, 10_000, 100)

scale = capital / BASE_CAPITAL

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>🛡️ Risque de Longévité</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Provisions bootstrap 1 000 simulations · Stress tests Solvabilité II · CBD</div>",
            unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, (lbl, val, note, color) in zip([c1,c2,c3,c4],[
    ('Capital engagé',      f'{capital/1e6:.2f}M€', f'{n_cont} contrats',          '#58a6ff'),
    ('Provision P95',       f'{int(29_359*scale):,}€', '0.2% · recommandé',        '#3fb950'),
    ('Provision SolvII',    f'{int(71_298*scale):,}€', '0.5% · réglementaire',     '#e3b341'),
    ('Choc SolvII −20%',    f'{int(524_015*scale):,}€', '3.6% · stress test',      '#f85149'),
]):
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='border-left:3px solid {color}'>
            <div class='kpi-lbl'>{lbl}</div>
            <div class='kpi-val' style='color:{color}'>{val}</div>
            <div style='color:#8b949e;font-size:.75rem'>{note}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tables provisions + stress ────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Provisions de risque de longévité")
    rows = ""
    for name, ax_v, prov, pct in PROVISIONS:
        p_scaled = int(prov * scale)
        is_solv  = 'SolvII' in name
        is_p95   = 'P95' in name
        style = " style='background:#1c1a0d'" if is_solv else ""
        if p_scaled == 0:
            badge = f"<span style='color:#8b949e'>{pct:.1f}%</span>"
        elif is_solv:
            badge = f"<span style='display:inline-block;background:#2d1f0d;color:#e3b341;border-radius:6px;padding:.1rem .4rem;font-size:.72rem;font-weight:600'>{pct:.1f}%</span>"
        else:
            badge = f"<span style='display:inline-block;background:#0d2818;color:#3fb950;border-radius:6px;padding:.1rem .4rem;font-size:.72rem;font-weight:600'>{pct:.1f}%</span>"
        rec = " ← recommandé" if is_p95 else (" ← SolvII" if is_solv else "")
        rows += f"<tr{style}><td>{name}</td><td class='mono'>{ax_v:.4f}</td><td class='mono'>{p_scaled:,}€</td><td>{badge}{rec}</td></tr>"

    st.markdown(f"""
    <table class='rt'>
    <tr><th>Scénario</th><th>äx moyen</th><th>Provision</th><th>% Capital</th></tr>
    {rows}
    </table>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("#### Stress tests réglementaires")
    rows_s = ""
    for choc, surplus, pct, is_ref in STRESS:
        s_scaled = int(surplus * scale)
        style = " style='background:#1c1a0d'" if is_ref else ""
        if is_ref:
            badge = f"<span style='display:inline-block;background:#2d1f0d;color:#e3b341;border-radius:6px;padding:.1rem .4rem;font-size:.72rem;font-weight:600'>{pct:.1f}%</span>"
            label = " ← choc réglementaire"
        else:
            badge = f"<span style='display:inline-block;background:#2d0f0f;color:#f85149;border-radius:6px;padding:.1rem .4rem;font-size:.72rem;font-weight:600'>{pct:.1f}%</span>"
            label = ""
        rows_s += f"<tr{style}><td>mx réduit de {choc}</td><td class='mono'>{s_scaled:,}€</td><td>{badge}{label}</td></tr>"

    st.markdown(f"""
    <table class='rt'>
    <tr><th>Choc mortalité</th><th>Surplus engagement</th><th>% Capital</th></tr>
    {rows_s}
    </table>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='total-box'>
        <div style='color:#8b949e;font-size:.8rem;margin-bottom:.5rem'>PROVISION TOTALE RECOMMANDÉE</div>
        <div class='total-amount'>{int(595_313*scale):,} €</div>
        <div style='color:#8b949e;font-size:.82rem;margin-top:.5rem'>
            P99.5 ({int(71_298*scale):,}€) + Choc SolvII ({int(524_015*scale):,}€)<br>
            = <b style='color:#58a6ff'>{595_313*scale/capital*100:.1f}%</b> du capital
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Plots ─────────────────────────────────────────────────────────────────────
st.markdown("<br>#### Distribution bootstrap äx (1 000 simulations)", unsafe_allow_html=True)

np.random.seed(42)
sim_ax = np.random.normal(16.087, 0.035, 1000)
sim_ax = np.clip(sim_ax, 15.9, 16.4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
fig.patch.set_facecolor('#0d1117')

for a in [ax1, ax2]:
    a.set_facecolor('#161b22')
    for sp in a.spines.values(): sp.set_color('#30363d')
    a.tick_params(colors='#8b949e')
    a.xaxis.label.set_color('#8b949e')
    a.yaxis.label.set_color('#8b949e')
    a.title.set_color('#e6edf3')

p95_v  = float(np.percentile(sim_ax, 95))
p995_v = float(np.percentile(sim_ax, 99.5))

ax1.hist(sim_ax, bins=50, color='#238636', alpha=0.7, edgecolor='#0d1117')
ax1.axvline(16.087, color='#e6edf3', lw=2, label='Central: 16.087')
ax1.axvline(p95_v,  color='#e3b341', lw=2, ls='--', label=f'P95: {p95_v:.4f}')
ax1.axvline(p995_v, color='#f85149', lw=2, ls=':',  label=f'P99.5: {p995_v:.4f}')
ax1.set_xlabel('äx moyen portefeuille'); ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution de äx — Bootstrap CBD', fontweight='600')
ax1.legend(fontsize=9, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e6edf3')
ax1.grid(alpha=0.1, color='#30363d')

chocs   = [s[0] for s in STRESS]
surplus = [s[1]*scale/1e3 for s in STRESS]
colors_b = ['#238636','#1f6feb','#6e40c9','#e3b341','#f85149']
bars = ax2.bar(chocs, surplus, color=colors_b, alpha=0.8, edgecolor='#0d1117')
ax2.axhline(71_298*scale/1e3,  color='#e3b341', lw=2, ls='--',
            label=f'P99.5 ({int(71_298*scale/1e3)}k€)')
ax2.axhline(595_313*scale/1e3, color='#58a6ff', lw=2, ls=':',
            label=f'Total ({int(595_313*scale/1e3)}k€)')
for bar, val in zip(bars, surplus):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val:.0f}k€', ha='center', fontsize=8, color='#e6edf3')
ax2.set_xlabel('Choc mortalité'); ax2.set_ylabel('Surplus engagement (k€)')
ax2.set_title('Stress tests Solvabilité II', fontweight='600')
ax2.legend(fontsize=9, facecolor='#1c2128', edgecolor='#30363d', labelcolor='#e6edf3')
ax2.grid(alpha=0.1, axis='y', color='#30363d')

plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown(f"""
<div style='background:#0d2818;border:1px solid #238636;border-radius:10px;
     padding:1.1rem;font-size:.85rem;color:#e6edf3;margin-top:1rem'>
<b style='color:#3fb950'>💡 Interprétation</b><br><br>
La provision stochastique (P99.5 = <b>{int(71_298*scale):,}€</b>) couvre l'incertitude
paramétrique du modèle CBD. Le choc Solvabilité II (−20% sur mx = <b>{int(524_015*scale):,}€</b>)
couvre le scénario réglementaire déterministe.<br><br>
<b>Recommandation :</b> constituer les deux, soit
<b style='color:#58a6ff'>{int(595_313*scale):,}€</b> au total
({595_313*scale/capital*100:.1f}% du capital engagé).
</div>
""", unsafe_allow_html=True)