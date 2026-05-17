#!/usr/bin/env python3
# ============================================================================
# 04b_patient_aware_benchmark.py  (PATCHED — tractable SVM)
#
# Same patient-aware CV logic as before, with one change:
#   training cells per fold are capped at MAX_TRAIN_CELLS to keep SVM tractable,
#   matching the 15,000-cell budget used in the original random-split benchmark.
#   The TEST set is never subsampled — every held-out cell/patient is scored.
#
# Set MAX_TRAIN_CELLS = None to disable subsampling (only do this if you have
# many hours per classifier and want to confirm subsampling doesn't change the
# story).
# ============================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MAX_TRAIN_CELLS = 15000        # cap training-set cells per fold; None disables
SUBSAMPLE_SEED  = 42           # reproducible subsample

for d in ('results', 'figures', 'supplementary'):
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("PATIENT-AWARE BENCHMARKING (addresses pseudo-replication)")
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Training cells per fold capped at: {MAX_TRAIN_CELLS}")
print("=" * 70)

# ----------------------------------------------------------------------------
# 1. Load atlases
# ----------------------------------------------------------------------------
MODEL_FILES = {
    'LinearSCVI':  'results/ovarian_LinearSCVI_atlas.h5ad',
    'Autoencoder': 'results/ovarian_Autoencoder_atlas.h5ad',
    'VAE':         'results/ovarian_VAE_atlas.h5ad',
}
LATENT_KEYS = {'LinearSCVI': 'X_scVI', 'Autoencoder': 'X_AE', 'VAE': 'X_VAE'}

atlases = {}
for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"  ✗ {name}: missing → skip"); continue
    atlases[name] = sc.read_h5ad(path)
    print(f"  ✓ {name}: {atlases[name].n_obs:,} cells × {atlases[name].n_vars:,} genes")

if not atlases:
    raise SystemExit("No atlases loaded — run training scripts first.")

# ----------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------
def make_clf(name, seed):
    if name == 'RF':    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == 'SVM':   return SVC(kernel='rbf', probability=True, random_state=seed)
    if name == 'LASSO': return LogisticRegression(penalty='l1', solver='liblinear',
                                                  random_state=seed, max_iter=1000)
    raise ValueError(name)

NEEDS_SCALING = {'RF': False, 'SVM': True, 'LASSO': True}
ML_MODELS     = ['RF', 'SVM', 'LASSO']

def cap_training_set(tr_idx, max_n, seed):
    """Random cap of training indices to keep SVM tractable."""
    if max_n is None or len(tr_idx) <= max_n:
        return tr_idx
    rng = np.random.default_rng(seed)
    return rng.choice(tr_idx, size=max_n, replace=False)

def aggregate_to_patient(prob, y, groups):
    df = pd.DataFrame({'patient': groups, 'prob': prob, 'true': y})
    g = df.groupby('patient').agg(prob=('prob', 'mean'), true=('true', 'first'))
    return g['prob'].to_numpy(), g['true'].to_numpy().astype(int)

# ----------------------------------------------------------------------------
# 3. GroupKFold
# ----------------------------------------------------------------------------
print("\n--- (A) GROUP K-FOLD (5 folds, split by patient_ID) ---")
gkf_records = []

for dl_name, adata in atlases.items():
    X = adata.obsm[LATENT_KEYS[dl_name]]
    y = (adata.obs['PFI_category_12_months'] == 'short').astype(int).to_numpy()
    groups = adata.obs['patient_ID'].astype(str).to_numpy()
    print(f"\n  {dl_name}: {len(y):,} cells | {len(np.unique(groups))} patients")

    for ml_name in ML_MODELS:
        cell_aucs, pat_aucs, pat_briers, pat_f1s = [], [], [], []
        gkf = GroupKFold(n_splits=5)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
            tr = cap_training_set(tr, MAX_TRAIN_CELLS,
                                  seed=SUBSAMPLE_SEED + fold)
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            gte = groups[te]

            if NEEDS_SCALING[ml_name]:
                s = StandardScaler().fit(Xtr)
                Xtr, Xte = s.transform(Xtr), s.transform(Xte)

            clf = make_clf(ml_name, seed=42)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]

            if len(np.unique(yte)) > 1:
                cell_aucs.append(roc_auc_score(yte, prob))
            p_prob, p_true = aggregate_to_patient(prob, yte, gte)
            if len(np.unique(p_true)) > 1:
                pat_aucs.append(roc_auc_score(p_true, p_prob))
                pat_briers.append(brier_score_loss(p_true, p_prob))
                pat_f1s.append(f1_score(p_true, (p_prob > 0.5).astype(int)))

        rec = {
            'DL_Model': dl_name, 'ML_Model': ml_name, 'CV_Strategy': 'GroupKFold-5',
            'Cell_AUC_mean':    np.mean(cell_aucs)  if cell_aucs  else np.nan,
            'Cell_AUC_std':     np.std(cell_aucs)   if cell_aucs  else np.nan,
            'Patient_AUC_mean': np.mean(pat_aucs)   if pat_aucs   else np.nan,
            'Patient_AUC_std':  np.std(pat_aucs)    if pat_aucs   else np.nan,
            'Patient_F1_mean':  np.mean(pat_f1s)    if pat_f1s    else np.nan,
            'Patient_Brier':    np.mean(pat_briers) if pat_briers else np.nan,
        }
        gkf_records.append(rec)
        print(f"    {ml_name:5s}: cell-AUC={rec['Cell_AUC_mean']:.3f}±{rec['Cell_AUC_std']:.3f}"
              f"   patient-AUC={rec['Patient_AUC_mean']:.3f}±{rec['Patient_AUC_std']:.3f}")

df_gkf = pd.DataFrame(gkf_records)
df_gkf.to_csv('results/Benchmark_GroupKFold_PatientAware.csv', index=False)
print("\n  ✓ Saved: results/Benchmark_GroupKFold_PatientAware.csv")

# ----------------------------------------------------------------------------
# 4. LOPO
# ----------------------------------------------------------------------------
print("\n--- (B) LEAVE-ONE-PATIENT-OUT ---")
lopo_records = []

for dl_name, adata in atlases.items():
    X = adata.obsm[LATENT_KEYS[dl_name]]
    y = (adata.obs['PFI_category_12_months'] == 'short').astype(int).to_numpy()
    groups = adata.obs['patient_ID'].astype(str).to_numpy()
    print(f"\n  {dl_name}")

    for ml_name in ML_MODELS:
        logo = LeaveOneGroupOut()
        preds = []
        for fold_i, (tr, te) in enumerate(logo.split(X, y, groups=groups)):
            tr = cap_training_set(tr, MAX_TRAIN_CELLS,
                                  seed=SUBSAMPLE_SEED + fold_i)
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            if NEEDS_SCALING[ml_name]:
                s = StandardScaler().fit(Xtr)
                Xtr, Xte = s.transform(Xtr), s.transform(Xte)
            clf = make_clf(ml_name, seed=42)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]
            preds.append({
                'patient':   groups[te][0],
                'mean_prob': float(prob.mean()),
                'true':      int(yte[0]),
                'n_cells':   int(len(yte)),
            })

        pdf = pd.DataFrame(preds)
        auc   = roc_auc_score(pdf['true'], pdf['mean_prob']) if pdf['true'].nunique() > 1 else np.nan
        brier = brier_score_loss(pdf['true'], pdf['mean_prob']) if pdf['true'].nunique() > 1 else np.nan
        f1    = f1_score(pdf['true'], (pdf['mean_prob'] > 0.5).astype(int))
        lopo_records.append({
            'DL_Model': dl_name, 'ML_Model': ml_name, 'CV_Strategy': 'LOPO',
            'Patient_AUC': auc, 'Patient_F1': f1, 'Patient_Brier': brier,
            'n_patients': len(pdf),
        })
        pdf.to_csv(f'results/LOPO_predictions_{dl_name}_{ml_name}.csv', index=False)
        print(f"    {ml_name:5s}: patient-AUC={auc:.3f}  F1={f1:.3f}  Brier={brier:.3f}")

df_lopo = pd.DataFrame(lopo_records)
df_lopo.to_csv('results/Benchmark_LOPO_PatientLevel.csv', index=False)
print("\n  ✓ Saved: results/Benchmark_LOPO_PatientLevel.csv")

# ----------------------------------------------------------------------------
# 5. Comparison with old random-split numbers (if Benchmark_Summary.csv exists)
# ----------------------------------------------------------------------------
old_path = 'results/Benchmark_Summary.csv'
if os.path.exists(old_path):
    old = pd.read_csv(old_path)[['DL_Model', 'ML_Model', 'AUC_mean', 'AUC_std']]
    old = old.rename(columns={'AUC_mean': 'RandomSplit_AUC_mean',
                              'AUC_std':  'RandomSplit_AUC_std'})
    merged = (old
              .merge(df_gkf[['DL_Model','ML_Model','Cell_AUC_mean','Cell_AUC_std',
                             'Patient_AUC_mean','Patient_AUC_std']],
                     on=['DL_Model','ML_Model'], how='right')
              .merge(df_lopo[['DL_Model','ML_Model','Patient_AUC']]
                       .rename(columns={'Patient_AUC':'LOPO_Patient_AUC'}),
                     on=['DL_Model','ML_Model'], how='left'))
    merged.to_csv('results/Benchmark_AllStrategies_Comparison.csv', index=False)
    print("  ✓ Saved comparison: results/Benchmark_AllStrategies_Comparison.csv")

# ----------------------------------------------------------------------------
# 6. Figure
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
plot_df = df_gkf[df_gkf['ML_Model'] == 'SVM'].copy()
xs = np.arange(len(plot_df)); w = 0.38
axes[0].bar(xs - w/2, plot_df['Cell_AUC_mean'], w, yerr=plot_df['Cell_AUC_std'],
            capsize=4, color='#377EB8', edgecolor='black', label='Cell-level')
axes[0].bar(xs + w/2, plot_df['Patient_AUC_mean'], w, yerr=plot_df['Patient_AUC_std'],
            capsize=4, color='#E41A1C', edgecolor='black', label='Patient-aggregated')
axes[0].axhline(0.5, color='gray', ls='--', alpha=0.6, label='Chance')
axes[0].set_xticks(xs); axes[0].set_xticklabels(plot_df['DL_Model'])
axes[0].set_ylabel('AUC'); axes[0].set_ylim(0, 1.05)
axes[0].set_title('GroupKFold-5, SVM\n(no patient leakage)', fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc='upper right', fontsize=9, framealpha=0.95)
axes[0].grid(alpha=0.3, axis='y')

sns.barplot(data=df_lopo, x='DL_Model', y='Patient_AUC', hue='ML_Model', ax=axes[1],
            palette=['#E41A1C', '#377EB8', '#4DAF4A'], edgecolor='black')
axes[1].axhline(0.5, color='gray', ls='--', alpha=0.6)
axes[1].set_ylabel('Patient-level AUC'); axes[1].set_ylim(0, 1.05)
axes[1].set_title('Leave-One-Patient-Out\n(clinical generalization)', fontweight='bold')
axes[1].set_ylim(0, 1.05)
axes[1].legend(title='Classifier', loc='upper right', fontsize=9,
               framealpha=0.95, ncol=3)
axes[1].grid(alpha=0.3, axis='y')
plt.savefig('figures/Figure_PatientAware_Benchmark.png', dpi=600, bbox_inches='tight')
plt.savefig('supplementary/Supplementary_Fig_PatientAware_Benchmark.png',
            dpi=600, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/Figure_PatientAware_Benchmark.png")

# ----------------------------------------------------------------------------
# 7. JSON summary
# ----------------------------------------------------------------------------
best_gkf  = df_gkf .loc[df_gkf ['Patient_AUC_mean'].idxmax()]
best_lopo = df_lopo.loc[df_lopo['Patient_AUC'].idxmax()]
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'config': {'max_train_cells_per_fold': MAX_TRAIN_CELLS},
    'best_under_groupkfold': {
        'DL_Model': best_gkf['DL_Model'], 'ML_Model': best_gkf['ML_Model'],
        'Cell_AUC':    f"{best_gkf['Cell_AUC_mean']:.3f} ± {best_gkf['Cell_AUC_std']:.3f}",
        'Patient_AUC': f"{best_gkf['Patient_AUC_mean']:.3f} ± {best_gkf['Patient_AUC_std']:.3f}",
    },
    'best_under_lopo': {
        'DL_Model':    best_lopo['DL_Model'], 'ML_Model': best_lopo['ML_Model'],
        'Patient_AUC': f"{best_lopo['Patient_AUC']:.3f}",
        'Brier':       f"{best_lopo['Patient_Brier']:.3f}",
        'n_patients':  int(best_lopo['n_patients']),
    },
}
with open('results/PatientAware_Benchmark_Summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("DONE.")
print(f"Best under GroupKFold: {best_gkf['DL_Model']}+{best_gkf['ML_Model']}  "
      f"cell-AUC={best_gkf['Cell_AUC_mean']:.3f}  patient-AUC={best_gkf['Patient_AUC_mean']:.3f}")
print(f"Best under LOPO:       {best_lopo['DL_Model']}+{best_lopo['ML_Model']}  "
      f"patient-AUC={best_lopo['Patient_AUC']:.3f}")
print("=" * 70)
