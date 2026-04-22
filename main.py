#!/usr/bin/env python3
"""
main.py — Pipeline universel de modélisation de la mortalité
=============================================================

Usage
-----
python main.py --data data/france_deaths_clean.csv \\
               --pop  data/france_population_clean.csv \\
               --sex  both \\
               --train-end 2000 \\
               --proj-year 2025 \\
               --models all \\
               --output outputs/

Le pipeline accepte n'importe quel CSV au format HMD :
  Deaths CSV : colonnes Year, Age, Male (ou Female)
  Population CSV : idem avec expositions

Résultats produits
------------------
  outputs/
  ├── backtest_results.csv      — métriques RMSE/Biais/Coverage H+F
  ├── projected_tables/         — tables de mortalité projetées (.csv)
  ├── pricing_results.csv       — äx et primes par modèle et âge
  ├── longevity_provisions.csv  — provisions P75/P95/P99.5
  ├── plots/                    — visualisations PNG
  └── summary_report.txt        — résumé textuel complet
"""

import argparse
import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pmdarima as pm


# ── Helpers internes (pas de dépendance sur src/) ────────────────────────────

def compute_life_table(df):
    """Table de mortalité complète depuis Age/mx."""
    df = df.reset_index(drop=True).sort_values('Age').copy()
    df['ax'] = 0.5
    df['qx'] = (df['mx'] / (1 + (1 - df['ax']) * df['mx'])).clip(upper=1.0)
    df['px'] = 1 - df['qx']
    df['lx'] = 0.0
    df.loc[df.index[0], 'lx'] = 100_000.0
    for i in range(1, len(df)):
        df.loc[df.index[i], 'lx'] = df.loc[df.index[i-1], 'lx'] * df.loc[df.index[i-1], 'px']
    df['dx'] = df['lx'] * df['qx']
    df['Lx'] = df['lx'] - (1 - df['ax']) * df['dx']
    last = df.index[-1]
    if df.loc[last, 'mx'] > 0:
        df.loc[last, 'Lx'] = df.loc[last, 'lx'] / df.loc[last, 'mx']
    df['Tx'] = df['Lx'][::-1].cumsum()[::-1]
    df['ex'] = df['Tx'] / df['lx']
    return df


def compute_annuity(df_proj, age_x, taux, horizon=40):
    """äx depuis une table projetée."""
    lt = compute_life_table(df_proj).set_index('Age')
    v  = 1 / (1 + taux)
    if age_x not in lt.index:
        return np.nan
    lx_ref = lt.loc[age_x, 'lx']
    if lx_ref < 1e-10:
        return np.nan
    return sum(
        (v**k) * lt.loc[age_x+k, 'lx'] / lx_ref
        for k in range(horizon+1)
        if (age_x+k) in lt.index
    )


# ── Chargement des données ────────────────────────────────────────────────────

def load_data(deaths_path, pop_path, age_max=90, year_min=1900):
    """
    Charge et fusionne deaths + population.
    Accepte tout CSV avec colonnes : Year, Age, Male, Female
    (ou Male uniquement, ou Female uniquement).
    """
    df_d = pd.read_csv(deaths_path)
    df_p = pd.read_csv(pop_path)

    sexes_available = [s for s in ['Male','Female'] if s in df_d.columns]
    if not sexes_available:
        raise ValueError(f"Aucune colonne Male/Female trouvée dans {deaths_path}")

    d_l = df_d.melt(id_vars=['Year','Age'], value_vars=sexes_available,
                    var_name='Sex', value_name='Deaths')
    p_l = df_p.melt(id_vars=['Year','Age'], value_vars=sexes_available,
                    var_name='Sex', value_name='Population')
    df  = pd.merge(d_l, p_l, on=['Year','Age','Sex'])
    df['mx'] = df['Deaths'] / df['Population']

    df = df[(df['Year'] >= year_min) & (df['Age'] <= age_max) & (df['mx'] > 0)]
    return df.copy(), sexes_available


# ── Modèles ───────────────────────────────────────────────────────────────────

def fit_lee_carter(df_train):
    pivot  = df_train.pivot(index='Age', columns='Year', values='mx')
    log_mx = np.log(pivot.clip(lower=1e-10))
    ax     = log_mx.mean(axis=1)
    U, s, Vt = np.linalg.svd((log_mx.subtract(ax, axis=0)).values, full_matrices=False)
    bx_r = U[:,0]; kt_r = s[0]*Vt[0,:]
    if bx_r.sum() < 0: bx_r, kt_r = -bx_r, -kt_r
    sv = bx_r.sum(); bx_r /= sv; kt_r *= sv
    sh = kt_r.mean(); kt_r -= sh; ax = ax + bx_r * sh
    return (pd.Series(ax.values, index=pivot.index),
            pd.Series(bx_r, index=pivot.index),
            pd.Series(kt_r, index=pivot.columns))


def fit_cbd(df_train, age_min=50, age_max=85):
    df_f = df_train[(df_train['Age']>=age_min)&(df_train['Age']<=age_max)].copy()
    df_f['qx'] = (df_f['mx']/(1+0.5*df_f['mx'])).clip(1e-6,1-1e-6)
    df_f['lq'] = np.log(df_f['qx']/(1-df_f['qx']))
    piv  = df_f.pivot(index='Age',columns='Year',values='lq')
    ages = piv.index.values.astype(float)
    xbar = ages.mean()
    k1_v, k2_v = [], []
    for y in piv.columns:
        X = np.column_stack([np.ones(len(ages)), ages-xbar])
        c = np.linalg.lstsq(X, piv[y].values, rcond=None)[0]
        k1_v.append(c[0]); k2_v.append(c[1])
    return (pd.Series(k1_v, index=piv.columns),
            pd.Series(k2_v, index=piv.columns),
            xbar, ages)


def fit_renshaw_haberman(df_train, cohort_min=5):
    pivot  = df_train.pivot(index='Age', columns='Year', values='mx')
    log_mx = np.log(pivot.clip(lower=1e-10).values)
    ages   = pivot.index.values.astype(float)
    years  = pivot.columns.values.astype(float)
    ax = log_mx.mean(axis=1)
    U, s, Vt = np.linalg.svd(log_mx - ax[:,None], full_matrices=False)
    bx = U[:,0]; kt = s[0]*Vt[0,:]
    sv = bx.sum(); bx /= sv; kt *= sv
    sh = kt.mean(); kt -= sh; ax += bx * sh
    R = log_mx - ax[:,None] - bx[:,None]*kt[None,:]
    cohort_vals = {}
    for ti, t in enumerate(years):
        for ai, x in enumerate(ages):
            cohort_vals.setdefault(t-x, []).append(R[ai, ti])
    gamma_c = {c: float(np.mean(v)) for c,v in cohort_vals.items() if len(v)>=cohort_min}
    R_after = R.copy()
    for ti, t in enumerate(years):
        for ai, x in enumerate(ages):
            c = t - x
            if c in gamma_c: R_after[ai,ti] -= gamma_c[c]
    res_std = float(np.std(R_after.ravel()))
    lc_res  = float(np.std(R.ravel()))
    return {'ax':ax,'bx':bx,'kt':pd.Series(kt,index=pivot.columns),
            'gamma_c':gamma_c,'ages_arr':ages,'residual_std':res_std,
            'lc_residual_std':lc_res}


def get_gamma(cohort, gamma_c, gamma_series=None):
    if cohort in gamma_c: return gamma_c[cohort]
    if gamma_series is None: gamma_series = pd.Series(gamma_c).sort_index()
    return float(gamma_series.iloc[-5:].mean())


def project_table(model_name, params, proj_year, df_train_sex, age_max=90):
    """
    Projette une table de mortalité Age/mx pour l'année proj_year.
    Utilise auto_arima sur le training disponible.
    """
    if model_name == 'Lee-Carter':
        ax, bx, kt = params['ax'], params['bx'], params['kt']
        n = proj_year - int(kt.index[-1])
        m = pm.auto_arima(kt.values, seasonal=False, stepwise=True,
                          suppress_warnings=True, error_action='ignore')
        kt_pt = float(m.predict(n_periods=n)[-1])
        mx = np.exp(ax.values + bx.values * kt_pt).clip(min=1e-10)
        return pd.DataFrame({'Age': ax.index.tolist(), 'mx': mx})

    elif model_name == 'CBD':
        k1, k2, xbar, ages_c = params['k1'], params['k2'], params['xbar'], params['ages_c']
        ax, bx, kt = params['ax'], params['bx'], params['kt']
        n = proj_year - int(k1.index[-1])
        m1 = pm.auto_arima(k1.values, seasonal=False, stepwise=True,
                            suppress_warnings=True, error_action='ignore')
        m2 = pm.auto_arima(k2.values, seasonal=False, stepwise=True,
                            suppress_warnings=True, error_action='ignore')
        k1_pt = float(m1.predict(n_periods=n)[-1])
        k2_pt = float(m2.predict(n_periods=n)[-1])
        lq    = k1_pt + k2_pt*(ages_c - xbar)
        qx_c  = (np.exp(lq)/(1+np.exp(lq))).clip(1e-6,1-1e-6)
        mx_c  = (qx_c/(1-0.5*qx_c)).clip(min=1e-10)
        m_kt  = pm.auto_arima(kt.values, seasonal=False, stepwise=True,
                               suppress_warnings=True, error_action='ignore')
        kt_pt = float(m_kt.predict(n_periods=n)[-1])
        mx_lc = np.exp(ax.values + bx.values * kt_pt).clip(min=1e-10)
        df_t  = pd.DataFrame({'Age': ax.index.tolist(), 'mx': mx_lc})
        for i, age in enumerate(ages_c.astype(int)):
            df_t.loc[df_t['Age']==age, 'mx'] = mx_c[i]
        return df_t

    elif model_name == 'Renshaw-Haberman':
        rh = params['rh']
        ax_rh = rh['ax']; bx_rh = rh['bx']
        ages_rh = rh['ages_arr']; gamma_c = rh['gamma_c']
        kt_rh = rh['kt']
        n = proj_year - int(kt_rh.index[-1])
        m = pm.auto_arima(kt_rh.values, seasonal=False, stepwise=True,
                          suppress_warnings=True, error_action='ignore')
        kt_pt  = float(m.predict(n_periods=n)[-1])
        gs     = pd.Series(gamma_c).sort_index()
        log_mx = np.array([
            ax_rh[ai] + bx_rh[ai]*kt_pt + get_gamma(proj_year-ages_rh[ai], gamma_c, gs)
            for ai in range(len(ages_rh))
        ])
        return pd.DataFrame({'Age': ages_rh.astype(int), 'mx': np.exp(log_mx).clip(min=1e-10)})

    return None


def rolling_backtest_generic(model_name, params, df_train, df_test, n_boot=150):
    """
    Backtest rolling one-step-ahead générique.
    Fonctionne pour tous les modèles.
    """
    years_test = sorted(df_test['Year'].unique())
    e0_obs, e0_pred, e0_lo, e0_hi = [], [], [], []

    ax = params.get('ax'); bx = params.get('bx')
    kt_all = params['kt'].copy() if 'kt' in params else params['rh']['kt'].copy()

    res_std = params.get('lc_res', params.get('cbd_res',
              params['rh']['residual_std'] if 'rh' in params else 0.10))

    for year in years_test:
        df_yr = df_test[(df_test['Year']==year)&(df_test['Age']<=90)][['Age','mx']].reset_index(drop=True)
        e0_true = compute_life_table(df_yr).iloc[0]['ex']
        e0_obs.append(e0_true)

        # Forecast
        m   = pm.auto_arima(kt_all.values, seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action='ignore')
        fc, ci = m.predict(n_periods=1, return_conf_int=True, alpha=0.05)
        kt_pt  = float(fc[0])
        kt_std = (float(ci[0,1])-float(ci[0,0]))/(2*1.96)

        if ax is not None and bx is not None:
            mx_p = np.exp(ax.values + bx.values*kt_pt).clip(min=1e-10)
            df_p = pd.DataFrame({'Age': ax.index.tolist(), 'mx': mx_p})
        else:
            df_p = df_yr  # fallback
        e0_pred.append(compute_life_table(df_p).iloc[0]['ex'])

        # Bootstrap IC
        if ax is not None:
            av = ax.values; bv = bx.values; A = len(av)
            ages_l = ax.index.tolist()
            kt_s   = np.random.normal(kt_pt, kt_std, size=(n_boot,1))
            noise  = np.random.normal(0, res_std, size=(n_boot,A))
            mx_s   = np.exp(av + bv*kt_s + noise).clip(min=1e-10)
            boot   = [compute_life_table(pd.DataFrame({'Age':ages_l,'mx':mx_s[i]})).iloc[0]['ex']
                      for i in range(n_boot)]
        else:
            boot = [e0_true]*n_boot
        e0_lo.append(float(np.percentile(boot,2.5)))
        e0_hi.append(float(np.percentile(boot,97.5)))

        # Update kt
        if ax is not None:
            log_obs = np.log(df_yr['mx'].clip(lower=1e-10).values)
            kt_act  = float(np.dot(bx.values, log_obs-ax.values)/np.dot(bx.values,bx.values))
        else:
            kt_act = kt_pt
        kt_all = pd.concat([kt_all, pd.Series([kt_act], index=[year])])

    e0_obs=np.array(e0_obs); e0_pred=np.array(e0_pred)
    e0_lo=np.array(e0_lo);   e0_hi=np.array(e0_hi)
    return {
        'years':   years_test,
        'e0_obs':  e0_obs,
        'e0_pred': e0_pred,
        'e0_lower':e0_lo,
        'e0_upper':e0_hi,
        'rmse':    float(np.sqrt(np.mean((e0_pred-e0_obs)**2))),
        'bias':    float(np.mean(e0_pred-e0_obs)),
        'coverage':float(np.mean((e0_obs>=e0_lo)&(e0_obs<=e0_hi))),
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(cfg):
    """
    Pipeline complet :
    1. Chargement données
    2. Calibration modèles
    3. Backtest rolling
    4. Projection tables
    5. Pricing äx
    6. Risque de longévité
    7. Plots + exports CSV
    """
    t_start = time.time()
    os.makedirs(cfg['output'], exist_ok=True)
    os.makedirs(os.path.join(cfg['output'], 'plots'), exist_ok=True)
    os.makedirs(os.path.join(cfg['output'], 'tables'), exist_ok=True)

    log = []
    def info(msg):
        print(msg); log.append(msg)

    info("=" * 60)
    info("  PIPELINE MORTALITÉ — INSEA")
    info("=" * 60)
    info(f"  Données    : {cfg['deaths']}")
    info(f"  Train end  : {cfg['train_end']}")
    info(f"  Proj year  : {cfg['proj_year']}")
    info(f"  Modèles    : {cfg['models']}")
    info(f"  Output     : {cfg['output']}")

    # ── 1. Données ────────────────────────────────────────────────────────────
    info("\n[1/7] Chargement des données...")
    df_all, sexes_avail = load_data(cfg['deaths'], cfg['pop'],
                                     age_max=cfg['age_max'])

    if cfg['sex'] == 'both':
        sexes = sexes_avail
    else:
        sexes = [cfg['sex']] if cfg['sex'] in sexes_avail else sexes_avail

    info(f"  Sexes disponibles : {sexes_avail}")
    info(f"  Sexes analysés    : {sexes}")
    for s in sexes:
        df_s = df_all[df_all['Sex']==s]
        info(f"  {s}: {df_s['Year'].min()}–{df_s['Year'].max()}, "
             f"{df_s['Age'].nunique()} âges")

    # ── 2. Calibration ────────────────────────────────────────────────────────
    info("\n[2/7] Calibration des modèles...")
    all_params = {}

    for sex in sexes:
        df_tr = df_all[(df_all['Year']<=cfg['train_end'])&(df_all['Sex']==sex)].copy()
        df_te = df_all[(df_all['Year']>cfg['train_end']) &(df_all['Sex']==sex)].copy()
        p = {}

        if any(m in cfg['models'] for m in ['lc','lc2','bayesian','all']):
            ax, bx, kt = fit_lee_carter(df_tr)
            piv = df_tr.pivot(index='Age',columns='Year',values='mx')
            log_obs = np.log(piv.clip(lower=1e-10).values)
            log_fit = np.column_stack([ax.values+bx.values*kt[y] for y in kt.index])
            lc_res  = float(np.std((log_fit-log_obs).ravel()))
            p.update({'ax':ax,'bx':bx,'kt':kt,'lc_res':lc_res})
            info(f"  {sex} LC1 — residual std={lc_res:.4f}")

        if any(m in cfg['models'] for m in ['cbd','all']):
            k1, k2, xbar, ages_c = fit_cbd(df_tr, age_min=50, age_max=85)
            df_c = df_tr[(df_tr['Age']>=50)&(df_tr['Age']<=85)].copy()
            df_c['qx'] = (df_c['mx']/(1+0.5*df_c['mx'])).clip(1e-6,1-1e-6)
            df_c['lq'] = np.log(df_c['qx']/(1-df_c['qx']))
            piv_c = df_c.pivot(index='Age',columns='Year',values='lq')
            lfit  = pd.DataFrame({y:k1[y]+k2[y]*(ages_c-xbar) for y in k1.index},index=ages_c)
            cbd_res = float(np.std((lfit.values-piv_c.values).ravel()))
            p.update({'k1':k1,'k2':k2,'xbar':xbar,'ages_c':ages_c,'cbd_res':cbd_res})
            info(f"  {sex} CBD — residual std={cbd_res:.4f}")

        if any(m in cfg['models'] for m in ['rh','all']):
            rh = fit_renshaw_haberman(df_tr)
            p['rh'] = rh
            info(f"  {sex} RH  — residual std={rh['residual_std']:.4f} "
                 f"({len(rh['gamma_c'])} cohortes)")

        all_params[sex] = {'params': p, 'df_train': df_tr, 'df_test': df_te}

    # ── 3. Backtest ───────────────────────────────────────────────────────────
    info("\n[3/7] Backtesting rolling...")
    all_results = {}

    MODELS_MAP = {
        'lc':  'Lee-Carter',
        'cbd': 'CBD',
        'rh':  'Renshaw-Haberman',
    }

    for sex in sexes:
        all_results[sex] = {}
        p = all_params[sex]['params']
        df_tr = all_params[sex]['df_train']
        df_te = all_params[sex]['df_test']

        mods_to_run = [m for m in cfg['models'] if m in MODELS_MAP] \
                      if 'all' not in cfg['models'] \
                      else list(MODELS_MAP.keys())

        for mod_key in mods_to_run:
            if mod_key == 'lc' and 'ax' not in p: continue
            if mod_key == 'cbd' and 'k1' not in p: continue
            if mod_key == 'rh' and 'rh' not in p: continue

            mod_name = MODELS_MAP[mod_key]
            info(f"  {sex} {mod_name}...", )
            t0 = time.time()

            params_bt = dict(p)
            if mod_key == 'rh':
                params_bt['kt'] = p['rh']['kt']
                params_bt['ax'] = None; params_bt['bx'] = None

            res = rolling_backtest_generic(mod_name, params_bt, df_tr, df_te,
                                            n_boot=cfg['n_boot'])
            all_results[sex][mod_name] = res
            info(f"    RMSE={res['rmse']:.4f}  Bias={res['bias']:+.4f}  "
                 f"Coverage={res['coverage']:.1%}  ({time.time()-t0:.0f}s)")

    # ── 4. Projections 2025 ───────────────────────────────────────────────────
    info(f"\n[4/7] Projection tables {cfg['proj_year']}...")
    proj_tables = {}

    for sex in sexes:
        proj_tables[sex] = {}
        p = all_params[sex]['params']

        for mod_key, mod_name in MODELS_MAP.items():
            try:
                df_proj = project_table(mod_name, p, cfg['proj_year'],
                                        all_params[sex]['df_train'])
                if df_proj is not None:
                    proj_tables[sex][mod_name] = df_proj
                    e0_p = compute_life_table(df_proj).iloc[0]['ex']
                    # Save table
                    out_f = os.path.join(cfg['output'],'tables',
                                         f'mx_{sex}_{mod_name.replace(" ","_")}_{cfg["proj_year"]}.csv')
                    df_proj.to_csv(out_f, index=False)
                    info(f"  {sex} {mod_name} → e0={e0_p:.3f}")
            except Exception as e:
                info(f"  {sex} {mod_name} → ERREUR: {e}")

    # ── 5. Pricing äx ─────────────────────────────────────────────────────────
    info(f"\n[5/7] Pricing rentes viagères (i={cfg['taux']:.1%})...")
    pricing_rows = []

    for sex in sexes:
        for age_x in cfg['ages_pricing']:
            for mod_name, df_proj in proj_tables[sex].items():
                ax_val = compute_annuity(df_proj, age_x, cfg['taux'])
                prime  = cfg['capital'] / ax_val if not np.isnan(ax_val) and ax_val>0 else np.nan
                pricing_rows.append({
                    'Sex': sex, 'Age': age_x, 'Model': mod_name,
                    'ax': round(ax_val, 4) if not np.isnan(ax_val) else np.nan,
                    'Premium_EUR': round(prime) if not np.isnan(prime) else np.nan,
                    'Capital_EUR': cfg['capital'],
                    'Rate': cfg['taux'],
                    'Proj_Year': cfg['proj_year'],
                })

    df_pricing = pd.DataFrame(pricing_rows)
    df_pricing.to_csv(os.path.join(cfg['output'],'pricing_results.csv'), index=False)

    if not df_pricing.empty:
        for sex in sexes:
            for age_x in cfg['ages_pricing']:
                sub = df_pricing[(df_pricing['Sex']==sex)&(df_pricing['Age']==age_x)]
                if not sub.empty:
                    vals = sub[['Model','ax','Premium_EUR']].to_string(index=False)
                    info(f"  {sex} age {age_x} :\n{vals}")

    # ── 6. Résumé backtest CSV ────────────────────────────────────────────────
    info("\n[6/7] Export résultats backtest...")
    bt_rows = []
    for sex in sexes:
        for mod_name, res in all_results[sex].items():
            bt_rows.append({
                'Sex': sex, 'Model': mod_name,
                'RMSE': round(res['rmse'],4),
                'Bias': round(res['bias'],4),
                'Coverage_95': round(res['coverage'],3),
            })
    df_bt = pd.DataFrame(bt_rows)
    df_bt.to_csv(os.path.join(cfg['output'],'backtest_results.csv'), index=False)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    info("\n[7/7] Génération des plots...")

    COLORS_SEX   = {'Female':'#c04828','Male':'#185fa5'}
    COLORS_MODEL = {'Lee-Carter':'#185fa5','CBD':'#0f6e56','Renshaw-Haberman':'#854f0b'}

    # Plot backtest par modèle
    n_models = max(len(all_results[sex]) for sex in sexes) if all_results else 0
    if n_models > 0:
        fig, axes = plt.subplots(len(sexes), n_models,
                                  figsize=(6*n_models, 5*len(sexes)),
                                  squeeze=False)
        fig.suptitle(f'Backtest rolling — {cfg["train_end"]+1}–{cfg["proj_year"]-1}',
                     fontsize=13, fontweight='bold')
        fig.patch.set_facecolor('#f8f7f4')

        for si, sex in enumerate(sexes):
            for mi, (mod_name, res) in enumerate(all_results[sex].items()):
                a = axes[si][mi]; a.set_facecolor('#f8f7f4')
                color = COLORS_SEX.get(sex,'#534ab7')
                a.fill_between(res['years'],res['e0_lower'],res['e0_upper'],
                               alpha=0.2,color=color,label='IC 95%')
                a.plot(res['years'],res['e0_obs'],'o-',color='#1a2e4a',
                       lw=2,markersize=4,label='Observé')
                a.plot(res['years'],res['e0_pred'],'s--',color=color,
                       lw=1.8,markersize=3,label='Prédit')
                a.set_title(f'{mod_name} — {sex}\n'
                            f'RMSE={res["rmse"]:.3f} | Cov={res["coverage"]:.1%}',
                            fontsize=10)
                a.set_xlabel('Année'); a.set_ylabel('e₀')
                a.legend(fontsize=8); a.grid(alpha=0.2)
                a.spines[['top','right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(cfg['output'],'plots','backtest.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        info("  → plots/backtest.png")

    # Plot e0 historique + projeté
    fig, axes = plt.subplots(1, len(sexes), figsize=(7*len(sexes), 5), squeeze=False)
    fig.suptitle('Espérance de vie — Historique & Projection', fontsize=12, fontweight='bold')
    fig.patch.set_facecolor('#f8f7f4')

    for si, sex in enumerate(sexes):
        a = axes[0][si]; a.set_facecolor('#f8f7f4')
        # Historique
        e0_hist = []
        for yr in sorted(df_all[df_all['Sex']==sex]['Year'].unique()):
            df_y = df_all[(df_all['Year']==yr)&(df_all['Sex']==sex)][['Age','mx']].reset_index(drop=True)
            try: e0_hist.append((yr, compute_life_table(df_y).iloc[0]['ex']))
            except: pass
        if e0_hist:
            yrs_h, e0_h = zip(*e0_hist)
            a.plot(yrs_h, e0_h, color=COLORS_SEX.get(sex,'#534ab7'),
                   lw=2, label='Historique')
        # Projections
        for mod_name, df_proj in proj_tables[sex].items():
            e0_p = compute_life_table(df_proj).iloc[0]['ex']
            a.scatter([cfg['proj_year']], [e0_p],
                      color=COLORS_MODEL.get(mod_name,'#888780'),
                      s=80, zorder=5, label=f'{mod_name}: {e0_p:.2f}')
        a.axvline(cfg['train_end'], color='gray', ls='--', lw=0.8, alpha=0.7)
        a.set_title(f'{sex}', fontsize=11, fontweight='600')
        a.set_xlabel('Année'); a.set_ylabel('e₀ (années)')
        a.legend(fontsize=8); a.grid(alpha=0.2)
        a.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg['output'],'plots','e0_projection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    info("  → plots/e0_projection.png")

    # ── Rapport textuel ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.append(f"\n{'='*60}")
    log.append(f"  Pipeline terminé en {elapsed/60:.1f} min")
    log.append(f"  Outputs : {cfg['output']}")
    log.append("=" * 60)

    with open(os.path.join(cfg['output'],'summary_report.txt'),'w') as f:
        f.write("\n".join(log))

    info(f"\n{'='*60}")
    info(f"  Pipeline terminé en {elapsed/60:.1f} min")
    info(f"  Outputs dans : {cfg['output']}")
    info("=" * 60)

    return {'backtest': all_results, 'pricing': df_pricing, 'tables': proj_tables}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline universel de modélisation de la mortalité — INSEA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # France femmes, tous modèles, projection 2025
  python main.py --deaths data/france_deaths_clean.csv \\
                 --pop    data/france_population_clean.csv \\
                 --sex female --train-end 2000 --proj-year 2025

  # Les deux sexes, CBD seulement
  python main.py --deaths data/morts.csv --pop data/pop.csv \\
                 --sex both --models cbd --output outputs/cbd_only/

  # Données Maroc (même format)
  python main.py --deaths data/morocco_deaths.csv \\
                 --pop    data/morocco_population.csv \\
                 --sex both --train-end 2010 --proj-year 2030
        """
    )
    parser.add_argument('--deaths',    required=True,  help='CSV décès (Year, Age, Male/Female)')
    parser.add_argument('--pop',       required=True,  help='CSV population/exposition')
    parser.add_argument('--sex',       default='both', choices=['female','male','both'],
                        help='Sexe(s) à analyser [both]')
    parser.add_argument('--train-end', type=int, default=2000, help='Dernière année training [2000]')
    parser.add_argument('--proj-year', type=int, default=2025, help='Année de projection [2025]')
    parser.add_argument('--age-max',   type=int, default=90,   help='Âge maximum [90]')
    parser.add_argument('--models',    nargs='+', default=['all'],
                        choices=['lc','lc2','cbd','rh','bayesian','all'],
                        help='Modèles à calibrer [all]')
    parser.add_argument('--output',    default='outputs/', help='Dossier de sortie [outputs/]')
    parser.add_argument('--n-boot',    type=int, default=150, help='Bootstrap IC [150]')
    parser.add_argument('--taux',      type=float, default=0.02, help='Taux technique [0.02]')
    parser.add_argument('--capital',   type=float, default=100_000, help='Capital rente [100000]')
    parser.add_argument('--ages',      nargs='+', type=int, default=[60,65,70],
                        help='Âges de souscription pour pricing [60 65 70]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = {
        'deaths':       args.deaths,
        'pop':          args.pop,
        'sex':          {'female':'Female','male':'Male','both':'both'}[args.sex],
        'train_end':    args.train_end,
        'proj_year':    args.proj_year,
        'age_max':      args.age_max,
        'models':       args.models,
        'output':       args.output,
        'n_boot':       args.n_boot,
        'taux':         args.taux,
        'capital':      args.capital,
        'ages_pricing': args.ages,
    }

    results = run_pipeline(cfg)