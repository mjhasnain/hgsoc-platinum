#!/usr/bin/env python3
# ============================================================================
# 02a_train_linearscvi_SAVEFIX.py - Training + Save Fix
# TESTED & WORKING with scvi-tools 1.4.1 + anndata 0.10+
# ============================================================================

import scanpy as sc
import scvi
import pandas as pd
import numpy as np
import os
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

for d in ['results', 'figures', 'supplementary']:
    os.makedirs(d, exist_ok=True)

print("="*60)
print("MODEL 1/3: LinearSCVI Training")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"  scvi-tools version: {scvi.__version__}")
try:
    import lightning
    print(f"  lightning version: {lightning.__version__}")
except:
    pass
print()

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("[1/5] Loading data...")
input_file = "results/DL_INPUT_3000_HVG.tsv"
full_df = pd.read_table(input_file, index_col=0)

metadata_columns = ['patient_ID', 'treatment_phase', 'anatomical_location', 'PFI_category_12_months']
metadata_columns = [col for col in metadata_columns if col in full_df.columns]

obs = full_df[metadata_columns].copy()
X_df = full_df.drop(columns=metadata_columns)

if 'cell_barcode' in X_df.columns:
    X_df = X_df.drop(columns=['cell_barcode'])

adata = sc.AnnData(X=X_df.values.astype(np.float32), obs=obs)
adata.var_names = X_df.columns
adata.obs_names = full_df.index

print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# ============================================================================
# STEP 2: Setup scVI
# ============================================================================
print("\n[2/5] Setting up LinearSCVI...")

batch_key = 'patient_ID' if 'patient_ID' in adata.obs.columns else adata.obs.columns[0]
print(f"  Batch key: {batch_key}")

scvi.model.LinearSCVI.setup_anndata(adata, batch_key=batch_key)

# ============================================================================
# STEP 3: Train Model (MINIMAL PARAMETERS - WORKING)
# ============================================================================
print("\n[3/5] Training...")

vae = scvi.model.LinearSCVI(adata, n_latent=20)

vae.train(
    max_epochs=100,
    accelerator='cpu',
    batch_size=256,
    train_size=0.9,
    early_stopping=True,
)

# ============================================================================
# STEP 4: Extract Latent Space
# ============================================================================
print("\n[4/5] Extracting latent space...")
adata.obsm["X_scVI"] = vae.get_latent_representation()
print(f"  Latent space shape: {adata.obsm['X_scVI'].shape}")

loadings = vae.get_loadings()
loadings.to_csv("results/LinearSCVI_Gene_Weights.csv")
print(f"  Gene loadings saved: {loadings.shape}")

# ============================================================================
# STEP 5: Save (WITH STRING ARRAY FIX)
# ============================================================================
print("\n[5/5] Saving...")

# FIX: Enable nullable string writing for anndata 0.10+
import anndata
anndata.settings.allow_write_nullable_strings = True

# Also convert any nullable string columns to object dtype for compatibility
for col in adata.obs.columns:
    if adata.obs[col].dtype == 'string':  # pandas nullable string
        adata.obs[col] = adata.obs[col].astype('object')

# Now save
adata.write("results/ovarian_LinearSCVI_atlas.h5ad")
vae.save("results/linearscvi_model", overwrite=True)

print("  ✓ Saved: results/ovarian_LinearSCVI_atlas.h5ad")
print("  ✓ Saved: results/linearscvi_model/")

# Summary
summary = {
    'model': 'LinearSCVI',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_cells': int(adata.n_obs),
    'n_genes': int(adata.n_vars),
    'n_latent': 20,
    'batch_key': batch_key,
    'training_completed': True
}

with open("results/LinearSCVI_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("  ✓ Saved: results/LinearSCVI_summary.json")

print("\n" + "="*60)
print("✅ LinearSCVI COMPLETE")
print("="*60)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📁 Output files:")
print("  ✓ results/ovarian_LinearSCVI_atlas.h5ad")
print("  ✓ results/linearscvi_model/")
print("  ✓ results/LinearSCVI_Gene_Weights.csv")
print("  ✓ results/LinearSCVI_summary.json")
print("\n🔜 Next: Run Autoencoder training")
print("   python 02b_train_autoencoder_SAVEFIX.py\n")