#!/usr/bin/env python3
# ============================================================================
# 04c_cross_dataset_validation.py
#
# WHY:  The strongest argument against "the model just learned patient/batch
#       identity" is to train on one independent dataset and test on the other.
#       Two different studies (GSE266577 vs GSE165897) = two different
#       tissue-collection sites, library prep protocols, and patient cohorts.
#       Signal that survives this is biological, not technical.
#
# WHAT IT DOES:
#   - Splits the combined atlas by `dataset` / GSE ID
#   - Trains on dataset A's cells, tests on dataset B's cells
#   - Reports cell-level AND patient-level AUC for both directions (A→B, B→A)
#   - Saves predictions and a figure suitable for a main paper supplement
#
# REQUIRES: the same trained atlases as 04b. Cells must carry an obs column
#   identifying which dataset they came from. If your atlas doesn't have one,
#   the script tries to infer it from `patient_ID` prefix; edit DATASET_KEY
#   or INFER_FROM_PID accordingly.
# ============================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, f1_score

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ----------------------------------------------------------------------------
# Config — edit if your column is named differently
# ----------------------------------------------------------------------------
DATASET_KEY = 'dataset'        # name of obs column holding GSE ID or 'A'/'B'
INFER_FROM_PID = True          # if DATASET_KEY not present, try to infer
                               # (treats first half of patient IDs as A, rest as B —
                               #  REPLACE with your real mapping below)

ATLASES = {
    'LinearSCVI':  ('results/ovarian_LinearSCVI_atlas.h5ad',  'X_scVI'),
    'Autoencoder': ('results/ovarian_Autoencoder_atlas.h5ad', 'X_AE'),
    'VAE':         ('results/ovarian_VAE_atlas.h5ad',         'X_VAE'),
}
CLASSIFIERS = ['SVM', 'RF', 'LASSO']

def make_clf(name, seed=42):
    if name == 'RF':    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == 'SVM':   return SVC(kernel='rbf', probability=True, random_state=seed)
    if name == 'LASSO': return LogisticRegression(penalty='l1', solver='liblinear',
                                                  random_state=seed, max_iter=1000)
NEEDS_SCALE = {'RF': False, 'SVM': True, 'LASSO': True}

# ----------------------------------------------------------------------------
def assign_dataset(adata):
    """Return a numpy array of dataset labels, one per cell."""
    if DATASET_KEY in adata.obs.columns:
        return adata.obs[DATASET_KEY].astype(str).to_numpy()
    if not INFER_FROM_PID:
        raise KeyError(f"No '{DATASET_KEY}' column and INFER_FROM_PID=False.")
    # EDIT THIS MAPPING with your real patient→dataset assignment:
    # e.g. GSE266577 patients vs GSE165897 patients
    pids = adata.obs['patient_ID'].astype(str)
    # Default heuristic — half/half by sorted unique IDs. REPLACE with truth.
    uniq = sorted(pids.unique())
    half = len(uniq) // 2
    mapping = {p: ('GSE266577' if i < half else 'GSE165897')
               for i, p in enumerate(uniq)}
    print("  ⚠ Inferred dataset from patient IDs — EDIT the mapping for truth.")
    return pids.map(mapping).to_numpy()

def aggregate_to_patient(prob, y, groups):
    df = pd.DataFrame({'patient': groups, 'prob': prob, 'true': y})
    g = df.groupby('patient').agg(prob=('prob', 'mean'), true=('true', 'first'))
    return g['prob'].to_numpy(), g['true'].to_numpy().astype(int)

# ----------------------------------------------------------------------------
print("=" * 72)
print("CROSS-DATASET VALIDATION (GSE266577 ⇄ GSE165897)")
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("=" * 72)

records = []
roc_curves = {}

for dl_name, (path, latent_key) in ATLASES.items():
    if not os.path.exists(path):
        print(f"\n  ✗ {dl_name}: atlas missing")
        continue
    print(f"\n--- {dl_name} ---")
    adata = sc.read_h5ad(path)
    X      = adata.obsm[latent_key]
    y      = (adata.obs['PFI_category_12_months'] == 'short').astype(int).to_numpy()
    pid    = adata.obs['patient_ID'].astype(str).to_numpy()
    ds     = assign_dataset(adata)
    datasets = sorted(np.unique(ds))
    if len(datasets) != 2:
        print(f"  ⚠ found {len(datasets)} datasets, expected 2 — skipping {dl_name}")
        continue
    A, B = datasets
    print(f"  Dataset A = {A}  ({(ds==A).sum():,} cells, "
          f"{len(np.unique(pid[ds==A]))} patients)")
    print(f"  Dataset B = {B}  ({(ds==B).sum():,} cells, "
          f"{len(np.unique(pid[ds==B]))} patients)")

    for direction, train_set, test_set in [('A→B', A, B), ('B→A', B, A)]:
        tr = ds == train_set
        te = ds == test_set
        # Skip if either side is single-class
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  {direction}: single-class on one side — skipping")
            continue

        for ml in CLASSIFIERS:
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            pte = pid[te]

            if NEEDS_SCALE[ml]:
                s = StandardScaler().fit(Xtr)
                Xtr, Xte = s.transform(Xtr), s.transform(Xte)

            clf = make_clf(ml)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]

            cell_auc = roc_auc_score(yte, prob)
            p_prob, p_true = aggregate_to_patient(prob, yte, pte)
            pat_auc = roc_auc_score(p_true, p_prob) if len(np.unique(p_true)) > 1 else np.nan
            pat_brier = brier_score_loss(p_true, p_prob) if len(np.unique(p_true)) > 1 else np.nan
            pat_f1  = f1_score(p_true, (p_prob > 0.5).astype(int))

            records.append({
                'DL_Model': dl_name, 'ML_Model': ml, 'Direction': direction,
                'Train_dataset': train_set, 'Test_dataset': test_set,
                'n_train_cells': int(tr.sum()), 'n_test_cells': int(te.sum()),
                'n_test_patients': int(len(np.unique(pte))),
                'Cell_AUC': cell_auc, 'Patient_AUC': pat_auc,
                'Patient_F1': pat_f1, 'Patient_Brier': pat_brier,
            })
            print(f"  {direction} {ml:5s}: cell-AUC={cell_auc:.3f}  "
                  f"patient-AUC={pat_auc:.3f}  F1={pat_f1:.3f}")

            if ml == 'SVM':
                fpr, tpr, _ = roc_curve(yte, prob)
                roc_curves[(dl_name, direction)] = (fpr, tpr, cell_auc)

df = pd.DataFrame(records)
df.to_csv('results/CrossDataset_Validation.csv', index=False)
print(f"\n  ✓ Saved: results/CrossDataset_Validation.csv ({len(df)} rows)")

# ----------------------------------------------------------------------------
# Figure: ROC for SVM in each direction, per DL model
# ----------------------------------------------------------------------------
if roc_curves:
    n = len({k[0] for k in roc_curves})
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    dl_order = [d for d in ATLASES if any(k[0] == d for k in roc_curves)]
    for i, dl in enumerate(dl_order):
        ax = axes[0, i]
        for direction, color in [('A→B', '#E41A1C'), ('B→A', '#377EB8')]:
            if (dl, direction) in roc_curves:
                fpr, tpr, auc = roc_curves[(dl, direction)]
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f'{direction}  AUC = {auc:.3f}')
        ax.plot([0,1], [0,1], 'k--', alpha=0.4)
        ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
        ax.set_title(dl, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    plt.suptitle('Cross-dataset validation (SVM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Figure_CrossDataset_ROC.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_CrossDataset_ROC.png")

print("\n" + "=" * 72)
print("DONE. Cross-dataset AUC is the strongest single number you can report.")
print("=" * 72)