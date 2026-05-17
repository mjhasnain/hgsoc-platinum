#!/usr/bin/env python3
# ============================================================================
# 03_compare_all_models.py - Compare All 3 DL Models
# ============================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("="*60)
print("MODEL COMPARISON: LinearSCVI vs Autoencoder vs VAE")
print("="*60)

# Load all 3 atlases
models = {
    'LinearSCVI': sc.read_h5ad(os.path.join(config['paths']['output_dir'], "ovarian_LinearSCVI_atlas.h5ad")),
    'Autoencoder': sc.read_h5ad(os.path.join(config['paths']['output_dir'], "ovarian_Autoencoder_atlas.h5ad")),
    'VAE': sc.read_h5ad(os.path.join(config['paths']['output_dir'], "ovarian_VAE_atlas.h5ad"))
}

results = []

for name, adata in models.items():
    print(f"\nEvaluating {name}...")
    
    # Get latent space
    if name == 'LinearSCVI':
        X_latent = adata.obsm["X_scVI"]
    elif name == 'Autoencoder':
        X_latent = adata.obsm["X_AE"]
    else:
        X_latent = adata.obsm["X_VAE"]
    
    # Target
    y = (adata.obs['PFI_category_12_months'] == 'short').astype(int).values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_latent, y, test_size=0.3, random_state=42)
    
    # Train RF
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'AUC': roc_auc_score(y_test, y_prob),
        'F1': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    })
    
    print(f"  AUC: {results[-1]['AUC']:.3f}")

# Save comparison
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(config['paths']['output_dir'], "DL_Model_Comparison.csv"), index=False)

# Plot comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df_results.melt(id_vars='Model'), x='Model', y='value', hue='variable')
plt.xticks(rotation=45)
plt.ylabel('Score')
plt.title('Deep Learning Model Comparison')
plt.savefig(os.path.join(config['paths']['figures_dir'], "DL_Model_Comparison.tiff"), dpi=600)

print("\n✅ COMPARISON COMPLETE")
print(f"Saved: DL_Model_Comparison.csv\n")
print(df_results.to_string(index=False))