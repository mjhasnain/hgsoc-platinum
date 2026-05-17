#!/usr/bin/env python3
# ============================================================================
# 10_negative_controls.py  (PATCHED — tractable runtime)
#
#
#   1. Caps training cells per fold at MAX_TRAIN_CELLS (same as 04b)
#   2. Defaults to LASSO instead of SVM. For a permutation null the only
#      thing that matters is using the SAME classifier on real and shuffled
#      labels; LASSO is ~100× faster than SVM with probability=True and
#      gives an essentially identical empirical p-value.
#
# Expected runtime on a CPU node: 20–40 minutes total for 100 permutations
# across 3 DL models.
#
# If you want to also confirm the result with SVM, set CLASSIFIER = 'SVM'
# and either drop N_PERMUTATIONS to ~20 or run overnight.
# ============================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
N_PERMUTATIONS  = 100        # 100 is comfortable for p<0.01 resolution
N_SPLITS        = 5
MAX_TRAIN_CELLS = 15000      # match 04b's training budget
CLASSIFIER      = 'LASSO'    # 'LASSO' (fast, recommended) | 'RF' | 'SVM'
SUBSAMPLE_SEED  = 42

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

ATLASES = {
    'LinearSCVI':  ('results/ovarian_LinearSCVI_atlas.h5ad',  'X_scVI'),
    'Autoencoder': ('results/ovarian_Autoencoder_atlas.h5ad', 'X_AE'),
    'VAE':         ('results/ovarian_VAE_atlas.h5ad',         'X_VAE'),
}

def make_clf(name, seed):
    if name == 'LASSO':
        return LogisticRegression(penalty='l1', solver='liblinear',
                                  random_state=seed, max_iter=1000)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == 'SVM':
        return SVC(kernel='rbf', probability=True, random_state=seed)
    raise ValueError(name)

NEEDS_SCALING = {'LASSO': True, 'RF': False, 'SVM': True}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def cap_training_set(tr_idx, max_n, seed):
    if max_n is None or len(tr_idx) <= max_n:
        return tr_idx
    rng = np.random.default_rng(seed)
    return rng.choice(tr_idx, size=max_n, replace=False)

def run_one_gkf(X, y, groups, clf_name=CLASSIFIER, base_seed=42):
    """One GroupKFold pass. Returns mean cell-level AUC across folds."""
    aucs = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        tr = cap_training_set(tr, MAX_TRAIN_CELLS, seed=base_seed + fold)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if NEEDS_SCALING[clf_name]:
            s = StandardScaler().fit(Xtr)
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        if len(np.unique(ytr)) < 2:
            continue
        clf = make_clf(clf_name, seed=base_seed)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        if len(np.unique(yte)) > 1:
            aucs.append(roc_auc_score(yte, prob))
    return float(np.mean(aucs)) if aucs else np.nan

def patient_level_shuffle(y, groups, rng):
    """Shuffle labels AT THE PATIENT LEVEL — every cell from a patient still
    shares one label, but patient→label assignment is permuted."""
    pid_to_label = {}
    for p in np.unique(groups):
        idx = np.where(groups == p)[0][0]
        pid_to_label[p] = y[idx]
    patients = list(pid_to_label.keys())
    labels   = [pid_to_label[p] for p in patients]
    shuffled = rng.permutation(labels)
    new_map  = dict(zip(patients, shuffled))
    return np.array([new_map[p] for p in groups])

# ----------------------------------------------------------------------------
print("=" * 72)
print("NEGATIVE CONTROL — patient-level label permutation test")
print(f"Classifier:     {CLASSIFIER}")
print(f"Permutations:   {N_PERMUTATIONS}")
print(f"Train cap/fold: {MAX_TRAIN_CELLS}")
print(f"Started:        {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 72)

results = []
null_distributions = {}

for dl_name, (path, latent_key) in ATLASES.items():
    if not os.path.exists(path):
        print(f"\n  ✗ {dl_name}: atlas missing"); continue
    print(f"\n--- {dl_name} ---  ({datetime.now():%H:%M:%S})")
    adata = sc.read_h5ad(path)
    X = adata.obsm[latent_key]
    y = (adata.obs['PFI_category_12_months'] == 'short').astype(int).to_numpy()
    g = adata.obs['patient_ID'].astype(str).to_numpy()

    # 1) Real labels
    real_auc = run_one_gkf(X, y, g, base_seed=42)
    print(f"  Real-label AUC = {real_auc:.3f}")

    # 2) Permutations
    null_aucs = []
    rng = np.random.default_rng(42)
    for k in range(N_PERMUTATIONS):
        y_perm = patient_level_shuffle(y, g, rng)
        if len(np.unique(y_perm)) < 2:
            continue
        auc = run_one_gkf(X, y_perm, g, base_seed=1000 + k)
        if not np.isnan(auc):
            null_aucs.append(auc)
        if (k + 1) % 10 == 0:
            print(f"    perm {k+1:3d}/{N_PERMUTATIONS}  "
                  f"null mean = {np.mean(null_aucs):.3f}  "
                  f"({datetime.now():%H:%M:%S})")
    null_aucs = np.array(null_aucs)
    p_emp = float((null_aucs >= real_auc).mean()) if len(null_aucs) else np.nan

    print(f"  Null AUC (mean ± sd): {null_aucs.mean():.3f} ± {null_aucs.std():.3f}")
    print(f"  Empirical p-value:    {p_emp:.4f}")

    results.append({
        'DL_Model': dl_name,
        'Classifier': CLASSIFIER,
        'Real_AUC': real_auc,
        'Null_AUC_mean': float(null_aucs.mean()) if len(null_aucs) else np.nan,
        'Null_AUC_std':  float(null_aucs.std())  if len(null_aucs) else np.nan,
        'Null_AUC_95_low':  float(np.percentile(null_aucs, 2.5))  if len(null_aucs) else np.nan,
        'Null_AUC_95_high': float(np.percentile(null_aucs, 97.5)) if len(null_aucs) else np.nan,
        'n_permutations_valid': int(len(null_aucs)),
        'empirical_p_value': p_emp,
    })
    null_distributions[dl_name] = (null_aucs, real_auc)

pd.DataFrame(results).to_csv('results/Negative_Control_Permutation.csv', index=False)
print("\n  ✓ Saved: results/Negative_Control_Permutation.csv")

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
if null_distributions:
    fig, axes = plt.subplots(1, len(null_distributions),
                             figsize=(5*len(null_distributions), 4.5), squeeze=False)
    for i, (dl, (null, real)) in enumerate(null_distributions.items()):
        ax = axes[0, i]
        ax.hist(null, bins=20, color='#999999', edgecolor='black', alpha=0.8,
                label=f'Null (n={len(null)})')
        ax.axvline(real, color='#E41A1C', lw=2.5,
                   label=f'Real labels AUC = {real:.3f}')
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('GroupKFold cell-level AUC')
        ax.set_ylabel('Frequency')
        ax.set_title(dl, fontweight='bold')
        ax.set_xlim(0.3, 1.0)
        # Add headroom so the legend doesn't sit on the leftmost bars
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.35)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

    plt.suptitle(f'Negative control: patient-level label permutation  ({CLASSIFIER})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Figure_Permutation_NullDistribution.png',
                dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_Permutation_NullDistribution.png")

print("\n" + "=" * 72)
print("DONE.")
print("Interpretation:")
print("  Real AUC > 95th percentile of null → signal is real biology (p<0.05).")
print("  Real AUC inside null distribution  → model was learning patient ID,")
print("  not PFI — would need stronger batch correction (e.g. scANVI).")
print("=" * 72)