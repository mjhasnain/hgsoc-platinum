#!/usr/bin/env python3
# ============================================================================
# 08_clinical_utility.py
#
#
# WHAT IT DOES (for each DL model × best classifier):
#   1. Load LOPO patient-level predictions
#   2. Calibration plot with 95% bootstrap CI (Hosmer-Lemeshow style bins)
#   3. Decision curve analysis (net benefit vs treat-all / treat-none)
#   4. Brier score with 1000-bootstrap CI
#   5. Save figures + a one-row-per-model CSV the discussion can cite
#
# REQUIRES:  run 04b_patient_aware_benchmark.py FIRST so the LOPO_predictions_*
#            CSVs exist.
# ============================================================================
import pandas as pd
import numpy as np
import os, glob, json
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Collect LOPO predictions
# ----------------------------------------------------------------------------
files = sorted(glob.glob('results/LOPO_predictions_*.csv'))
if not files:
    raise SystemExit("No LOPO_predictions_*.csv files. Run 04b first.")

frames = []
for f in files:
    name = os.path.basename(f).replace('LOPO_predictions_', '').replace('.csv', '')
    dl, ml = name.split('_')
    df = pd.read_csv(f)
    df['DL_Model'] = dl; df['ML_Model'] = ml
    frames.append(df)
preds_all = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(preds_all)} prediction rows across {preds_all['DL_Model'].nunique()} "
      f"DL models × {preds_all['ML_Model'].nunique()} classifiers")

# ----------------------------------------------------------------------------
# 2. Pick best ML per DL by patient-level AUC
# ----------------------------------------------------------------------------
def auc_for(df):
    if df['true'].nunique() < 2: return np.nan
    return roc_auc_score(df['true'], df['mean_prob'])

best = (preds_all.groupby(['DL_Model','ML_Model'])
                 .apply(auc_for).reset_index(name='Patient_AUC'))
best = best.loc[best.groupby('DL_Model')['Patient_AUC'].idxmax()]
print("\nBest classifier per DL model (by patient AUC):")
print(best.to_string(index=False))

# ----------------------------------------------------------------------------
# 3. Bootstrap helpers
# ----------------------------------------------------------------------------
def bootstrap_metric(y, p, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_pts = len(y)
    out = []
    for _ in range(n):
        idx = rng.integers(0, n_pts, n_pts)
        if len(np.unique(y[idx])) < 2 and fn is roc_auc_score:
            continue
        try: out.append(fn(y[idx], p[idx]))
        except Exception: pass
    return np.array(out)

def net_benefit(y, p, threshold):
    """DCA net benefit for a given threshold probability."""
    n = len(y)
    treat = p >= threshold
    tp = ((treat == 1) & (y == 1)).sum()
    fp = ((treat == 1) & (y == 0)).sum()
    if threshold >= 1: return 0.0
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

def nb_treat_all(y, threshold):
    n = len(y); pos = y.sum()
    if threshold >= 1: return 0.0
    return (pos / n) - ((n - pos) / n) * (threshold / (1 - threshold))

# ----------------------------------------------------------------------------
# 4. Build figure: calibration (top row) + DCA (bottom row), one column per DL model
# ----------------------------------------------------------------------------
dls = best['DL_Model'].tolist()
fig, axes = plt.subplots(2, len(dls), figsize=(5*len(dls), 9), squeeze=False)

clinical_records = []

for col, dl in enumerate(dls):
    ml = best.loc[best['DL_Model'] == dl, 'ML_Model'].iloc[0]
    df = preds_all[(preds_all['DL_Model']==dl) & (preds_all['ML_Model']==ml)].copy()
    y = df['true'].to_numpy().astype(int)
    p = df['mean_prob'].to_numpy()

    # --- Calibration (top row) ---
    ax = axes[0, col]
    n_bins = min(5, max(3, len(y) // 6))   # adaptive for small n
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy='quantile')
        brier = brier_score_loss(y, p)
        brier_boot = bootstrap_metric(y, p, brier_score_loss)
        brier_ci = (np.percentile(brier_boot, 2.5), np.percentile(brier_boot, 97.5))
        ax.plot(mean_pred, frac_pos, 'o-', color='#377EB8', lw=2, markersize=8,
            label=f'{dl} + {ml}\nBrier = {brier:.3f}  '
                  f'[{brier_ci[0]:.3f}, {brier_ci[1]:.3f}]')
        # (delete the ax.text(...) Brier annotation entirely)
    except Exception as e:
        ax.text(0.5, 0.5, f'calibration failed:\n{e}', ha='center', transform=ax.transAxes)

    ax.plot([0,1],[0,1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction (short PFI)')
    ax.set_title(f'Calibration: {dl} + {ml}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    brier = brier_score_loss(y, p)
    brier_boot = bootstrap_metric(y, p, brier_score_loss)
    brier_ci = (np.percentile(brier_boot, 2.5), np.percentile(brier_boot, 97.5))
    ax.text(0.04, 0.92, f'Brier = {brier:.3f}\n95% CI [{brier_ci[0]:.3f}, {brier_ci[1]:.3f}]',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))

    # --- DCA (bottom row) ---
    ax = axes[1, col]
    thresholds = np.linspace(0.01, 0.99, 99)
    nb_model    = [net_benefit(y, p, t) for t in thresholds]
    nb_all      = [nb_treat_all(y, t)   for t in thresholds]
    nb_none     = [0.0] * len(thresholds)
    ax.plot(thresholds, nb_model, color='#E41A1C', lw=2, label=f'{dl} + {ml}')
    ax.plot(thresholds, nb_all,   color='#999999', lw=1.5, linestyle='--', label='Treat all')
    ax.plot(thresholds, nb_none,  color='black',   lw=1, label='Treat none')
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'Decision curve: {dl} + {ml}', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    # Clip y-axis: model lines live in 0–0.7; the Treat-all curve crashing
    # at high threshold is what stretches auto-scale to -30 and crushes
    # everything informative into a thin band.
    ax.set_ylim(-0.15, max(0.7, max(nb_model) * 1.2))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)


    # Compute net benefit at clinically plausible thresholds
    nb_at = {}
    for t in [0.2, 0.3, 0.5, 0.7]:
        nb_at[f'NB_at_{t}'] = net_benefit(y, p, t)

    auc_val = auc_for(df)
    auc_boot = bootstrap_metric(y, p, roc_auc_score)
    auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5)) \
             if len(auc_boot) else (np.nan, np.nan)

    clinical_records.append({
        'DL_Model': dl, 'ML_Model': ml,
        'Patient_AUC': auc_val,
        'Patient_AUC_95_CI_low':  auc_ci[0],
        'Patient_AUC_95_CI_high': auc_ci[1],
        'Brier_score': brier,
        'Brier_95_CI_low':  brier_ci[0],
        'Brier_95_CI_high': brier_ci[1],
        **nb_at,
        'n_patients': len(y),
        'n_positive': int(y.sum()),
    })

plt.suptitle('Patient-level clinical utility (Leave-One-Patient-Out)',
             fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('figures/Figure_Clinical_Utility.png', dpi=600, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: figures/Figure_Clinical_Utility.png")

pd.DataFrame(clinical_records).to_csv('results/Clinical_Utility_Summary.csv', index=False)
print("  ✓ Saved: results/Clinical_Utility_Summary.csv")

print("\n" + "=" * 72)
print("DONE.  Cite these CIs in the discussion. The Brier score belongs in")
print("the abstract if calibration is reasonable; otherwise mention it as a")
print("limitation and explain why patient-level calibration is hard at n=32.")
print("=" * 72)