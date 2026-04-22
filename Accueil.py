"""
app.py — Dashboard Mortality Analytics
Streamlit multi-pages avec vrais résultats du projet
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

DEATHS_PATH = os.path.join(DATA_DIR, "france_deaths_clean.csv")
POP_PATH    = os.path.join(DATA_DIR, "france_population_clean.csv")

import streamlit as st

st.set_page_config(
    page_title="Mortality Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Page background ── */
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #58a6ff !important; font-size: 0.8rem !important; }

/* ── Hero ── */
.hero-card {
    background: linear-gradient(135deg, #1c2b3a 0%, #0d2137 50%, #0f2a1e 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #238636, #1f6feb, #58a6ff);
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: #e6edf3;
    margin: 0 0 0.5rem; line-height: 1.2;
}
.hero-sub { font-size: 0.95rem; color: #8b949e; margin: 0; }
.hero-tags { margin-top: 1rem; }
.tag {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 500;
    margin: 0.2rem; border: 1px solid;
}
.tag-green  { background: #0d2818; color: #3fb950; border-color: #238636; }
.tag-blue   { background: #0d1e36; color: #58a6ff; border-color: #1f6feb; }
.tag-purple { background: #1c1236; color: #bc8cff; border-color: #6e40c9; }
.tag-orange { background: #2d1f0d; color: #e3b341; border-color: #9e6a03; }

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem 1rem;
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
    border-radius: 12px 0 0 12px;
}
.kpi-card.green::after  { background: #238636; }
.kpi-card.blue::after   { background: #1f6feb; }
.kpi-card.purple::after { background: #6e40c9; }
.kpi-card.orange::after { background: #9e6a03; }

.kpi-label { font-size: 0.7rem; color: #8b949e; font-weight: 500;
             text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #e6edf3;
             font-family: 'DM Mono', monospace; margin: 0.3rem 0 0.1rem; line-height: 1; }
.kpi-delta { font-size: 0.78rem; font-weight: 500; }
.delta-green  { color: #3fb950; }
.delta-blue   { color: #58a6ff; }
.delta-orange { color: #e3b341; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 1.5rem 0 1rem;
}
.section-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #238636; flex-shrink: 0;
}
.section-title {
    font-size: 1rem; font-weight: 600; color: #e6edf3;
    margin: 0;
}

/* ── Result table ── */
.results-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.85rem; background: #161b22;
    border-radius: 10px; overflow: hidden;
    border: 1px solid #21262d;
}
.results-table th {
    background: #0d1117; color: #8b949e; padding: 0.7rem 1rem;
    text-align: left; font-weight: 500; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.06em;
    border-bottom: 1px solid #21262d;
}
.results-table td {
    padding: 0.65rem 1rem; color: #e6edf3;
    border-bottom: 1px solid #161b22;
}
.results-table tr:hover td { background: #1c2128; }
.results-table tr:last-child td { border-bottom: none; }
.best-val { color: #3fb950; font-weight: 600; font-family: 'DM Mono', monospace; }
.mono { font-family: 'DM Mono', monospace; }
.badge {
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 6px; font-size: 0.72rem; font-weight: 600;
}
.badge-green  { background: #0d2818; color: #3fb950; }
.badge-blue   { background: #0d1e36; color: #58a6ff; }
.badge-orange { background: #2d1f0d; color: #e3b341; }
.badge-purple { background: #1c1236; color: #bc8cff; }
.badge-red    { background: #2d0f0f; color: #f85149; }

/* ── Info boxes ── */
.info-box {
    background: #0d2818; border: 1px solid #238636;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.85rem; color: #e6edf3; margin: 0.5rem 0;
}
.warn-box {
    background: #2d1f0d; border: 1px solid #9e6a03;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.85rem; color: #e6edf3; margin: 0.5rem 0;
}

/* ── Nav card ── */
.nav-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 1.2rem;
    margin-bottom: 0.7rem; transition: border-color 0.2s;
}
.nav-card:hover { border-color: #1f6feb; }
.nav-card h4 { color: #e6edf3; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.3rem; }
.nav-card p  { color: #8b949e; font-size: 0.82rem; margin: 0; line-height: 1.5; }

/* ── Provision highlight ── */
.provision-box {
    background: linear-gradient(135deg, #0d2818, #0d1e36);
    border: 1px solid #1f6feb; border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.provision-amount {
    font-size: 2.4rem; font-weight: 700; color: #58a6ff;
    font-family: 'DM Mono', monospace;
}
.provision-label { font-size: 0.85rem; color: #8b949e; margin-top: 0.3rem; }

/* ── Footer ── */
.footer {
    text-align: center; color: #484f58; font-size: 0.75rem;
    padding: 2rem 0 1rem; border-top: 1px solid #21262d; margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0;border-bottom:1px solid #30363d;margin-bottom:1rem'>
        <div style='font-size:.95rem;font-weight:600;color:#e6edf3'>Mortality Analytics</div>
        <div style='font-size:.72rem;color:#8b949e;margin-top:.2rem'>INSEA </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.82rem;line-height:2.2;color:#e6edf3'>
    🏠 <b>Accueil</b><br>
    📊 Exploration des données<br>
    🛡️ Risque de Longévité<br>
    🔬 Comparaison des modèles<br>
    💰 Pricing rente viagère
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-card'>
    <div class='hero-title'>Modélisation Stochastique<br>de l'Espérance de Vie</div>
    <div class='hero-sub'>
        Comparaison de 5 modèles actuariels · Backtest rolling H+F 2001–2020 ·
        Pricing rente viagère · Quantification du risque de longévité
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='kpi-grid'>
    <div class='kpi-card green'>
        <div class='kpi-label'>Meilleur RMSE</div>
        <div class='kpi-value'>0.188</div>
        <div class='kpi-delta delta-green'>CBD Femmes · 0.228 Hommes</div>
    </div>
    <div class='kpi-card blue'>
        <div class='kpi-label'>Prime centrale F65</div>
        <div class='kpi-value'>5 558€</div>
        <div class='kpi-delta delta-blue'>/an · Capital 100k€ · i=2%</div>
    </div>
    <div class='kpi-card purple'>
        <div class='kpi-label'>Provision SolvII</div>
        <div class='kpi-value'>595k€</div>
        <div class='kpi-delta delta-blue'>4.1% capital · 100 rentes</div>
    </div>
    <div class='kpi-card orange'>
        <div class='kpi-label'>e₀ France F 2022</div>
        <div class='kpi-value'>85.7</div>
        <div class='kpi-delta delta-orange'>ans · H: 80.1 · Écart: 5.6</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Résultats backtest ────────────────────────────────────────────────────────
st.markdown("""
<div class='section-header'>
    <div class='section-dot'></div>
    <div class='section-title'>Backtest rolling one-step-ahead · France H+F · 2001–2020</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<table class='results-table'>
<tr>
    <th>Modèle</th>
    <th>RMSE ♀</th><th>Biais ♀</th><th>Cov. ♀</th>
    <th>RMSE ♂</th><th>Biais ♂</th><th>Cov. ♂</th>
    <th>Usage</th>
</tr>
<tr>
    <td><span class='badge badge-blue'>Lee-Carter</span></td>
    <td class='mono'>0.463</td><td class='mono'>+0.148</td>
    <td><span class='badge badge-orange'>82.6%</span></td>
    <td class='mono'>0.636</td><td class='mono'>−0.552</td>
    <td><span class='badge badge-red'>47.8%</span></td>
    <td style='color:#8b949e;font-size:0.8rem'>Référence historique</td>
</tr>
<tr>
    <td><span class='badge badge-purple'>LC2</span></td>
    <td class='mono'>0.601</td><td class='mono'>+0.374</td>
    <td><span class='badge badge-orange'>78.3%</span></td>
    <td class='mono'>0.556</td><td class='mono'>−0.441</td>
    <td><span class='badge badge-orange'>69.6%</span></td>
    <td style='color:#8b949e;font-size:0.8rem'>Prévision H améliorée</td>
</tr>
<tr>
    <td><span class='badge badge-green'>CBD ★</span></td>
    <td class='best-val'>0.188</td><td class='mono'>+0.084</td>
    <td><span class='badge badge-green'>91.3%</span></td>
    <td class='best-val'>0.228</td><td class='mono'>−0.087</td>
    <td><span class='badge badge-green'>95.7%</span></td>
    <td style='color:#3fb950;font-size:0.8rem;font-weight:600'>★ Tarification H+F</td>
</tr>
<tr>
    <td><span class='badge badge-orange'>Renshaw-Haberman</span></td>
    <td class='mono'>0.384</td><td class='best-val'>+0.052</td>
    <td><span class='badge badge-green'>87.0%</span></td>
    <td class='mono'>0.562</td><td class='mono'>−0.471</td>
    <td><span class='badge badge-red'>60.9%</span></td>
    <td style='color:#8b949e;font-size:0.8rem'>Réserves long terme</td>
</tr>
<tr>
    <td><span class='badge badge-purple'>LC Bayésien KF</span></td>
    <td class='mono'>0.482</td><td class='mono'>+0.139</td>
    <td class='best-val'>95.7%</td>
    <td class='mono'>0.692</td><td class='mono'>−0.624</td>
    <td><span class='badge badge-orange'>65.2%</span></td>
    <td style='color:#8b949e;font-size:0.8rem'>IC Solvabilité II ♀</td>
</tr>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Provisions + Pricing ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot' style='background:#6e40c9'></div>
        <div class='section-title'>Risque de longévité · 100 rentes · 14.37 M€</div>
    </div>
    <table class='results-table'>
    <tr><th>Scénario</th><th>Provision</th><th>% Capital</th></tr>
    <tr><td>Central (P50)</td><td class='mono'>0 €</td><td class='mono'>0.0%</td></tr>
    <tr><td>Sévère (P95)</td><td class='mono'>29 359 €</td><td class='mono'>0.2%</td></tr>
    <tr><td>SolvII (P99.5)</td><td class='mono'>71 298 €</td><td class='mono'>0.5%</td></tr>
    <tr><td style='color:#f85149'>Choc −20% (SolvII)</td>
        <td class='mono' style='color:#f85149'>524 015 €</td>
        <td class='mono' style='color:#f85149'>3.6%</td></tr>
    <tr style='background:#0d1e36'>
        <td><b style='color:#58a6ff'>Total recommandé</b></td>
        <td><b class='mono' style='color:#58a6ff'>595 313 €</b></td>
        <td><b class='mono' style='color:#58a6ff'>4.1%</b></td></tr>
    </table>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot' style='background:#1f6feb'></div>
        <div class='section-title'>Pricing rente viagère · 2025 · i=2%</div>
    </div>
    <table class='results-table'>
    <tr><th>Modèle</th><th>äx F65</th><th>Prime F</th><th>äx H65</th><th>Prime H</th></tr>
    <tr>
        <td><span class='badge badge-blue'>Lee-Carter</span></td>
        <td class='mono'>17.994</td><td class='mono'>5 558€</td>
        <td class='mono'>15.847</td><td class='mono'>6 312€</td>
    </tr>
    <tr>
        <td><span class='badge badge-green'>CBD</span></td>
        <td class='mono'>17.969</td><td class='mono'>5 565€</td>
        <td class='mono'>15.623</td><td class='mono'>6 401€</td>
    </tr>
    <tr>
        <td><span class='badge badge-orange'>RH</span></td>
        <td class='mono'>18.037</td><td class='mono'>5 544€</td>
        <td class='mono'>15.894</td><td class='mono'>6 293€</td>
    </tr>
    <tr style='background:#0d2818'>
        <td><b style='color:#3fb950'>Écart max</b></td>
        <td colspan='2' class='mono' style='color:#3fb950'>±21€ (0.4%)</td>
        <td colspan='2' class='mono' style='color:#3fb950'>±108€ (1.7%)</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

# ── Navigation pages ──────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:.6rem;margin:1.5rem 0 1rem'>
    <div style='width:8px;height:8px;border-radius:50%;background:#e3b341'></div>
    <div style='font-size:1rem;font-weight:600;color:#e6edf3'>Explorer le dashboard</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

for col, (icon, title, desc, page) in zip([c1, c2, c3, c4], [
    ("📊", "Exploration",
     "Trajectoires e₀ H+F, heatmap log(mx), écart Femmes–Hommes 1950–2024.",
     "pages/1_Exploration.py"),
    ("🛡️", "Risque de Longévité",
     "Provisions P95/P99.5, stress tests Solvabilité II, bootstrap 1 000 sim.",
     "pages/2_Longevite.py"),
    ("🔬", "Comparaison des modèles",
     "Résidus Lee-Carter, paramètres CBD κ1/κ2, tableau backtest H+F.",
     "pages/3_Modeles.py"),
    ("💰", "Pricing rente viagère",
     "Calculateur interactif äx, distribution VaR, prime prudentielle.",
     "pages/4_Pricing.py"),
]):
    with col:
        st.markdown(f"""
        <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;
             padding:1.2rem;margin-bottom:.5rem'>
            <div style='font-size:1.4rem;margin-bottom:.5rem'>{icon}</div>
            <div style='font-weight:600;color:#e6edf3;font-size:.95rem;
                        margin-bottom:.4rem'>{title}</div>
            <div style='color:#8b949e;font-size:.82rem;line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)
        st.page_link(page, label=f"Ouvrir →")

# ── Résultats backtesting ─────────────────────────────────────────────────────
# ── Outputs notebooks ─────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:.6rem;margin:1.5rem 0 1rem'>
    <div style='width:8px;height:8px;border-radius:50%;background:#3fb950'></div>
    <div style='font-size:1rem;font-weight:600;color:#e6edf3'>
        Résultats — Backtesting H+F & Analyse
    </div>
</div>
""", unsafe_allow_html=True)

import os
OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

plots = [
    ("outputs/backtest_HF_all_models.png",  "Backtest rolling — tous modèles H+F 2001–2020"),
    ("outputs/lc_residuals_HF.png",       "Résidus Lee-Carter — Carte de chaleur H vs F"),
    ("outputs/cbd_k1k2_HF.png",           "Paramètres CBD κ1/κ2 — Hommes vs Femmes"),
    ("outputs/gender_gap_e0.png",         "Espérance de vie & Écart H/F — 1950–2024"),
    ("outputs/gender_gap_decomposition..png",    "Décomposition de l'écart e₀ par âge"),
]

for filename, caption in plots:
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(full_path):
        st.markdown(f"<p style='color:#8b949e;font-size:.85rem;margin:.8rem 0 .3rem'>{caption}</p>",
                    unsafe_allow_html=True)
        st.image(full_path, use_container_width=True)
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#484f58;font-size:.75rem;
     padding:2rem 0 1rem;border-top:1px solid #21262d;margin-top:3rem'>
    Human Mortality Database · France 1950–2024 ·
    Lee-Carter · CBD · Renshaw-Haberman · Kalman Filter<br>
    INSEA 2024
</div>
""", unsafe_allow_html=True)