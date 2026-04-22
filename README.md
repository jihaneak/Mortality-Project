# Modélisation Stochastique de l'Espérance de Vie
### Stochastic Modelling of Life Expectancy & Longevity Risk Quantification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.54-red?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Models-5-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Data-HMD%20France%201950--2024-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square" />
</p>

<p align="center">
  <a href="https://mortality-project-x7frugeh3rmdappvmf4y9qq.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" />
  </a>
</p>

---

## 🇫🇷 Présentation

Ce projet propose une **comparaison rigoureuse de 5 modèles actuariels de projection de la mortalité** sur données françaises Hommes & Femmes (1950–2024), avec application à la **quantification du risque de longévité** dans un portefeuille de rentes viagères.

**Problème résolu :** Pour un portefeuille de 100 rentes représentant **14,37 M€** de capital engagé, quelle provision de longévité un assureur doit-il constituer pour rester solvable sous Solvabilité II ?

**Réponse :** **595 313 € (4,1% du capital)**, combinant une provision stochastique P99,5 (71 298 €) et un choc déterministe Solvabilité II de −20% sur la mortalité (524 015 €).

---

## 🇬🇧 Overview

This project presents a **systematic comparison of 5 actuarial mortality projection models** on French male and female data (1950–2024), applied to **longevity risk quantification** in a life annuity portfolio.

**Problem solved:** For a portfolio of 100 annuity contracts representing **€14.37M** in committed capital, what longevity provision must an insurer hold to remain solvent under Solvency II?

**Answer:** **€595,313 (4.1% of capital)**, combining a stochastic P99.5 provision (€71,298) and a Solvency II deterministic stress of −20% on mortality rates (€524,015).

---

## 📊 Dashboard interactif / Interactive Dashboard

🔗 **[mortality-project-x7frugeh3rmdappvmf4y9qq.streamlit.app](https://mortality-project-x7frugeh3rmdappvmf4y9qq.streamlit.app/)**

| Page | Contenu |
|------|---------|
| 🏠 Accueil | KPIs, résultats synthèse, navigation |
| 📊 Exploration | Trajectoires e₀, heatmap log(mx), écart H/F |
| 🔬 Modèles | Résidus Lee-Carter, paramètres CBD, backtests |
| 💰 Pricing | Calculateur äx interactif, VaR bootstrap |
| 🛡️ Longévité | Provisions P95/P99.5, stress tests Solvabilité II |

---

## 🔬 Modèles implémentés / Models

| Modèle | RMSE ♀ | RMSE ♂ | Coverage ♀ | Coverage ♂ | Usage |
|--------|--------|--------|------------|------------|-------|
| Lee-Carter (LC1) | 0.463 | 0.636 | 82.6% | 47.8% | Référence |
| Lee-Carter 2F (LC2) | 0.601 | 0.556 | 78.3% | 69.6% | Prévision H |
| **CBD** ★ | **0.188** | **0.228** | **91.3%** | **95.7%** | **Tarification** |
| Renshaw-Haberman | 0.384 | 0.562 | 87.0% | 60.9% | Réserves |
| LC Bayésien (KF) | 0.482 | 0.692 | 95.7% | 65.2% | Solvabilité II |

*Backtest rolling one-step-ahead · Période de validation : 2001–2020*

---

## 💰 Pricing rente viagère / Life Annuity Pricing

Femme 65 ans · Capital 100 000 € · i = 2% · Horizon 2025

| Modèle | äx | Prime annuelle |
|--------|-----|----------------|
| Lee-Carter | 17.994 | 5 558 €/an |
| CBD | 17.969 | 5 565 €/an |
| Renshaw-Haberman | 18.037 | 5 544 €/an |
| **Écart max** | **0.068** | **±21 €/an (0.4%)** |

> Remarquable convergence des 3 modèles : **écart < 0.4%** sur la prime annuelle.

---

## 🏗️ Architecture du projet / Project Structure

```
Mortality-Project/
├── Accueil.py                    # Dashboard principal (Streamlit)
├── pages/
│   ├── 1_Exploration.py          # Exploration des données
│   ├── 2_Longevite.py            # Risque de longévité
│   ├── 3_Modeles.py              # Comparaison des modèles
│   └── 4_Pricing.py              # Pricing rente viagère
├── src/
│   └── models/
│       ├── lee_carter.py         # Modèle Lee-Carter (SVD)
│       ├── cbd_model.py          # Modèle CBD (logit-âge)
│       ├── renshaw_haberman.py   # Modèle RH (effet cohorte)
│       ├── kalman_filter.py      # Extension bayésienne (KF)
│       ├── life_expectancy.py    # Table de mortalité
│       ├── uncertainty.py        # Bootstrap IC (3 sources)
│       ├── forecast_evaluation.py # Backtest rolling
│       └── pricing.py            # Rente viagère äx
├── notebooks/
│   ├── backtesting_HF.ipynb      # Backtest H+F tous modèles
│   ├── lc2_model.ipynb           # Lee-Carter 2 facteurs
│   └── longevity_risk.ipynb      # Quantification risque
├── data/
│   ├── france_deaths_clean.csv   # HMD décès France 1816–2024
│   └── france_population_clean.csv
├── outputs/                      # Plots notebooks
├── main.py                       # Pipeline CLI universel
└── requirements.txt
```

---

## ⚙️ Pipeline universel / Universal Pipeline

Ce projet inclut un pipeline CLI réutilisable pour **n'importe quelle donnée au format HMD** :

```bash
# France Hommes & Femmes, tous modèles
python main.py --deaths data/france_deaths_clean.csv \
               --pop    data/france_population_clean.csv \
               --sex both --train-end 2000 --proj-year 2025 \
               --models all --output outputs/

# Données d'un autre pays (même format)
python main.py --deaths data/autre_pays_deaths.csv \
               --pop    data/autre_pays_population.csv \
               --sex both --train-end 2015 --proj-year 2035
```

**Outputs automatiques :**
- `backtest_results.csv` — RMSE / Biais / Coverage
- `pricing_results.csv` — äx et primes par âge
- `tables/` — tables de mortalité projetées
- `plots/` — visualisations

---

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/jihaneak/Mortality-Project.git
cd Mortality-Project

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run Accueil.py
```

---

## 📦 Stack technique / Tech Stack

| Composant | Outil |
|-----------|-------|
| Langage | Python 3.12 |
| Dashboard | Streamlit 1.54 |
| Modélisation | numpy · pandas · statsmodels |
| ARIMA | pmdarima (auto_arima) |
| Optimisation MLE | scipy.optimize (Nelder-Mead) |
| Visualisation | matplotlib |
| Données | Human Mortality Database (HMD) |

---

## 📐 Méthodologie / Methodology

### Table de mortalité
```
mx → qx → lx → dx → Lx → Tx → ex
qx = mx / (1 + (1−ax)·mx)    [hypothèse UDD]
ex = Tx / lx
```

### Lee-Carter
```
log(mx,t) = ax + bx·kt + εxt
Calibration SVD · Contraintes : Σbx=1, Σkt=0
Projection kt : ARIMA(p,d,q) via auto_arima
```

### CBD
```
logit(qx,t) = κ1(t) + κ2(t)·(x − x̄)
Ages 50–85 · OLS par année · Projection indépendante κ1, κ2
```

### Renshaw-Haberman
```
log(mx,t) = ax + bx·kt + γ(t−x) + εxt
γc estimé par moyenne cohortale des résidus Lee-Carter
Réduction résidu std : 0.0975 → 0.0748 (−23.3%)
```

### Kalman Filter bayésien
```
État  : kt = kt−1 + δ + ηt,    ηt ~ N(0, σ²proc)
Obs   : kt_obs = kt + εt,       εt ~ N(0, σ²obs)
σproc=1.548, σobs=1.353 estimés par MLE (Nelder-Mead)
```

---

## 📈 Résultats clés / Key Results

- **CBD** : meilleur RMSE H+F (0.188 ♀ / 0.228 ♂), seul modèle avec coverage ≥ 90% pour les deux sexes
- **LC2** : améliore le coverage masculin de +21.8 points vs Lee-Carter (47.8% → 69.6%)
- **RH** : meilleur biais (+0.052 ans), réduction variance inexpliquée de 23.3%
- **Convergence pricing** : écart < 0.4% entre modèles sur äx
- **Provision totale** : 595 313 € = 4.1% du capital pour 100 rentes (14.37 M€)

---

## 📄 Rapport scientifique / Scientific Report

Le projet inclut un rapport complet en LaTeX, disponible en deux versions :
- 🇫🇷 **Version française** — style *Insurance: Mathematics and Economics*
- 🇬🇧 **English version** — JEL Classification: G22, J11, C53, C11

---

## 👤 Auteur / Author

**Akharaz Jihane**
2ème année — Institut National de Statistique et d'Économie Appliquée (INSEA)
Rabat, Maroc

[![GitHub](https://img.shields.io/badge/GitHub-jihaneak-black?style=flat-square&logo=github)](https://github.com/jihaneak)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-FF4B4B?style=flat-square&logo=streamlit)](https://mortality-project-x7frugeh3rmdappvmf4y9qq.streamlit.app/)

---

## 📚 Références / References

- Lee, R.D., & Carter, L.R. (1992). *Modeling and forecasting U.S. mortality.* JASA, 87(419).
- Cairns, A.J.G., Blake, D., & Dowd, K. (2006). *A two-factor model for stochastic mortality.* Journal of Risk and Insurance, 73(4).
- Renshaw, A.E., & Haberman, S. (2006). *A cohort-based extension to the Lee-Carter model.* Insurance: Mathematics and Economics, 38(3).
- Kalman, R.E. (1960). *A new approach to linear filtering.* Journal of Basic Engineering, 82(1).
- Human Mortality Database (2024). UC Berkeley & Max Planck Institute. [mortality.org](https://www.mortality.org)

---

*Données : Human Mortality Database · France 1950–2024 · Hommes & Femmes*