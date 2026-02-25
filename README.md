# Mortality Modeling and Forecasting Framework

End-to-end demographic and actuarial pipeline for estimating mortality, modeling age–time dynamics, and forecasting life expectancy using Human Mortality Database–type data.

**Live dashboard:** [\[Streamlit URL\]](https://mortality-project-ihskfdca3baphkxnb7gjgf.streamlit.app/)

---

## What this project does

- Computes mortality rates from deaths and exposures (Poisson framework)
- Builds complete actuarial life tables
- Quantifies statistical uncertainty (confidence intervals, Delta method)
- Fits and forecasts the **Lee–Carter** mortality model
- Projects future life expectancy
- Provides interactive visualization via Streamlit

---

## Data

Place files in: data/


The pipeline is general and can be applied to any country.

---

## Methods

- Poisson mortality model  
- Life table construction with flexible \(a_x\) assumptions (including Coale–Demeny adjustments)  
- Lee–Carter model
- Random walk with drift for \(k_t\) forecasting  
- Monte Carlo simulation for life expectancy uncertainty  

---

## Run
Install dependencies: pip install -r requirements.txt

Run full pipeline: python src/main.py
git
Launch dashboard streamlit run app.py


---

## Applications

Longevity risk • Life insurance • Pension modeling • Demographic forecasting • Health analytics

---

**Author**  
Jihane Akharaz – INSEA (Institut National de Statistique et d'Economie appliquée)
