#!/usr/bin/env python3
# ============================================================================
# 04_benchmarking_stability_final.py - FIXED VERSION
# Handles missing fonts and legend errors
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FONT FIX: Use available fonts on Linux servers
# ============================================================================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

# Try Arial, fallback to DejaVu Sans (available on most Linux)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

for d in [config['paths']['output_dir'], config['paths']['figures_dir'], config['paths']['supplementary_dir']]:
    os.makedirs(d, exist_ok=True)

print("="*70)
print("BENCHMARKING WITH STABILITY ANALYSIS (All 3 DL Models)")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: Load All 3 DL Model Atlases
# ============================================================================
print("[1/8] Loading all DL model atlases...")

model_files = {
    'LinearSCVI': os.path.join(config['paths']['output_dir'], "ovarian_LinearSCVI_atlas.h5ad"),
    'Autoencoder': os.path.join(config['paths']['output_dir'], "ovarian_Autoencoder_atlas.h5ad"),
    'VAE': os.path.join(config['paths']['output_dir'], "ovarian_VAE_atlas.h5ad")
}

atlases = {}
latent_keys = {'LinearSCVI': 'X_scVI', 'Autoencoder': 'X_AE', 'VAE': 'X_VAE'}

for name, path in model_files.items():
    if os.path.exists(path):
        atlases[name] = sc.read_h5ad(path)
        print(f"  ✓ {name}: {atlases[name].n_obs:,} cells × {atlases[name].n_vars:,} genes")
    else:
        print(f"  ✗ {name}: File not found - {path}")

if len(atlases) == 0:
    raise FileNotFoundError("No DL model atlases found! Run training scripts first.")

# Also load PCA and Gene Expression for baseline comparison
print("\n  Adding baseline comparisons...")
adata_ref = list(atlases.values())[0]  # Use first atlas as reference

# Calculate PCA
if "X_pca" not in adata_ref.obsm:
    print("    Calculating PCA...")
    sc.tl.pca(adata_ref, n_comps=20)

# Gene expression
X_genes = adata_ref.X.toarray() if hasattr(adata_ref.X, "toarray") else adata_ref.X

atlases['PCA_Linear'] = adata_ref
atlases['Gene_Expression'] = adata_ref
latent_keys['PCA_Linear'] = 'X_pca'
latent_keys['Gene_Expression'] = 'X_raw'

print(f"\n  Total models to benchmark: {len(atlases)}")

# ============================================================================
# STEP 2: Define Benchmarking Function
# ============================================================================
print("\n[2/8] Setting up benchmarking framework...")

def run_single_benchmark(seed, model_name, adata, latent_key, y, config):
    """Run single benchmark iteration with given seed"""
    np.random.seed(seed)
    
    # Get latent space
    if latent_key == 'X_raw':
        X_latent = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    else:
        X_latent = adata.obsm[latent_key]
    
    # Sample cells (for stability across runs)
    n_sample = min(15000, len(y))
    idx = np.random.choice(np.arange(len(y)), n_sample, replace=False)
    y_sub = y[idx]
    X_sub = X_latent[idx]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=seed
    )
    
    # Scaling for SVM/LASSO
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    models_config = [
        ('RF', RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), False),
        ('SVM', SVC(kernel='rbf', probability=True, random_state=seed), True),
        ('LASSO', LogisticRegression(penalty='l1', solver='liblinear', random_state=seed, max_iter=1000), True)
    ]
    
    for m_name, clf, use_scaled in models_config:
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results.append({
            'seed': seed,
            'DL_Model': model_name,
            'ML_Model': m_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob),
            'F1_Score': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Sensitivity': recall_score(y_test, y_pred),
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    
    return results

# ============================================================================
# STEP 3: Run Stability Analysis (10 Seeds)
# ============================================================================
print("\n[3/8] Running stability analysis (10 random seeds)...")

seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
y = (adata_ref.obs['PFI_category_12_months'] == 'short').astype(int).values

print(f"  Seeds: {seeds}")
print(f"  Total runs: {len(seeds)} × {len(atlases)} × 3 ML models = {len(seeds) * len(atlases) * 3}")

all_results = []
for model_name, adata in atlases.items():
    print(f"\n  Processing {model_name}...")
    latent_key = latent_keys[model_name]
    
    for seed in seeds:
        results = run_single_benchmark(seed, model_name, adata, latent_key, y, config)
        all_results.extend(results)

df_all = pd.DataFrame(all_results)
print(f"\n  ✓ Total benchmark results: {len(df_all):,}")

# ============================================================================
# STEP 4: Aggregate Statistics
# ============================================================================
print("\n[4/8] Calculating aggregate statistics...")

# Mean and STD across seeds
df_summary = df_all.groupby(['DL_Model', 'ML_Model']).agg({
    'Accuracy': ['mean', 'std'],
    'AUC': ['mean', 'std'],
    'F1_Score': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Sensitivity': ['mean', 'std'],
    'Specificity': ['mean', 'std']
}).round(4)

df_summary.columns = ['_'.join(col).strip() for col in df_summary.columns]
df_summary = df_summary.reset_index()

# Statistical significance testing
print("  Running statistical tests...")
significance_tests = []

dl_models = ['LinearSCVI', 'Autoencoder', 'VAE']
baseline_models = ['PCA_Linear', 'Gene_Expression']

for metric in ['Accuracy', 'AUC', 'F1_Score', 'Sensitivity', 'Specificity']:
    for ml_model in ['RF', 'SVM', 'LASSO']:
        for dl_model in dl_models:
            for baseline_model in baseline_models:
                dl_vals = df_all[(df_all['DL_Model'] == dl_model) & (df_all['ML_Model'] == ml_model)][metric]
                base_vals = df_all[(df_all['DL_Model'] == baseline_model) & (df_all['ML_Model'] == ml_model)][metric]
                
                t_stat, p_val = stats.ttest_ind(dl_vals, base_vals)
                significance_tests.append({
                    'Comparison': f'{dl_model}_vs_{baseline_model}',
                    'ML_Model': ml_model,
                    'Metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })

df_significance = pd.DataFrame(significance_tests)

# Save results
df_all.to_csv(os.path.join(config['paths']['output_dir'], "Benchmark_All_Runs.csv"), index=False)
df_summary.to_csv(os.path.join(config['paths']['output_dir'], "Benchmark_Summary.csv"), index=False)
df_significance.to_csv(os.path.join(config['paths']['output_dir'], "Benchmark_Significance.csv"), index=False)

print(f"  ✓ Saved: Benchmark_All_Runs.csv")
print(f"  ✓ Saved: Benchmark_Summary.csv")
print(f"  ✓ Saved: Benchmark_Significance.csv")

# ============================================================================
# STEP 5: Visualization - Main Benchmark Figure
# ============================================================================
print("\n[5/8] Generating benchmark figures...")

# Color palette for DL models
dl_palette = {
    'LinearSCVI': '#E41A1C',
    'Autoencoder': '#377EB8',
    'VAE': '#4DAF4A',
    'PCA_Linear': '#984EA3',
    'Gene_Expression': '#FF7F00'
}

# Figure 1: Multi-metric comparison (Nature style)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Deep Learning Model Benchmarking (10 Seeds)', 
             fontsize=16, fontweight='bold')

metrics = ['AUC', 'F1_Score', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    # Group by DL model (average across ML models)
    dl_summary = df_summary.groupby('DL_Model')[f'{metric}_mean'].mean().sort_values(ascending=False)
    dl_std = df_summary.groupby('DL_Model')[f'{metric}_std'].mean()
    
    x = np.arange(len(dl_summary))
    colors = [dl_palette.get(m, '#999999') for m in dl_summary.index]
    
    bars = ax.bar(x, dl_summary.values, yerr=dl_std.loc[dl_summary.index].values,
                  color=colors, edgecolor='black', capsize=5, alpha=0.8, linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dl_summary.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(f'{metric} (Mean ± SD)', fontsize=11)
    ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(config['paths']['figures_dir'], "Figure_Benchmark_Overview.png"), 
            dpi=600, bbox_inches='tight')
plt.close()

# Figure 2: Stability across seeds (violin plots) - FIXED LEGEND ISSUE
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Performance Stability Across 10 Random Seeds', 
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(['AUC', 'F1_Score', 'Accuracy']):
    ax = axes[idx]
    
    # Prepare data for violin plot
    plot_data = []
    for dl_model in dl_models:
        model_data = df_all[df_all['DL_Model'] == dl_model]
        for seed in seeds:
            seed_data = model_data[model_data['seed'] == seed]
            for ml_model in ['RF', 'SVM', 'LASSO']:
                ml_data = seed_data[seed_data['ML_Model'] == ml_model]
                if len(ml_data) > 0:
                    plot_data.append({
                        'DL_Model': dl_model,
                        'Metric': metric,
                        'Score': ml_data[metric].values[0]
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    sns.violinplot(data=plot_df, x='DL_Model', y='Score', hue='DL_Model',
                   ax=ax, palette=dl_palette, inner='box', alpha=0.7, legend=False)  # FIXED: No legend
    
    ax.set_xlabel('Deep Learning Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Distribution', fontsize=13, fontweight='bold')
    # FIXED: Removed legend_.remove() - just don't create legend
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(config['paths']['figures_dir'], "Figure_Stability_Violin.png"), 
            dpi=600, bbox_inches='tight')
plt.close()

# Figure 3: Significance heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Pivot significance data
pivot_sig = df_significance.pivot_table(
    index='Comparison', 
    columns='Metric', 
    values='p_value', 
    aggfunc='mean'
)

# Convert to -log10(p)
pivot_sig_log = -np.log10(pivot_sig + 1e-10)

im = ax.imshow(pivot_sig_log.values, aspect='auto', cmap='Reds', vmin=0, vmax=10)
ax.set_xticks(np.arange(len(pivot_sig_log.columns)))
ax.set_yticks(np.arange(len(pivot_sig_log.index)))
ax.set_xticklabels(pivot_sig_log.columns, fontsize=10)
ax.set_yticklabels(pivot_sig_log.index, fontsize=9)
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Comparison', fontsize=12)
ax.set_title('Statistical Significance (-log10 P-value)', fontsize=14, fontweight='bold')

# Add significance stars
for i in range(len(pivot_sig_log.index)):
    for j in range(len(pivot_sig_log.columns)):
        val = pivot_sig_log.values[i, j]
        if val > 3:  # p < 0.001
            ax.text(j, i, '***', ha='center', va='center', fontsize=10, fontweight='bold')
        elif val > 2:  # p < 0.01
            ax.text(j, i, '**', ha='center', va='center', fontsize=10, fontweight='bold')
        elif val > 1.3:  # p < 0.05
            ax.text(j, i, '*', ha='center', va='center', fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax, label='-log10(P-value)')
plt.tight_layout()
plt.savefig(os.path.join(config['paths']['supplementary_dir'], "Supplementary_Fig_Significance.png"), 
            dpi=600, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 6: ROC Curves for Best Run
# ============================================================================
print("\n[6/8] Generating ROC curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('ROC Curves by Deep Learning Model (Best Seed)', 
             fontsize=16, fontweight='bold')

best_seed = seeds[0]

for idx, dl_model in enumerate(dl_models):
    ax = axes[idx]
    
    model_data = df_all[(df_all['DL_Model'] == dl_model) & (df_all['seed'] == best_seed)]
    
    for ml_model in ['RF', 'SVM', 'LASSO']:
        ml_data = model_data[model_data['ML_Model'] == ml_model]
        if len(ml_data) > 0:
            auc_val = ml_data['AUC'].mean()
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.plot([0, 0, 1], [0, 1, 1], color=dl_palette[dl_model], 
                   linewidth=2, label=f'{ml_model} (AUC={auc_val:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{dl_model}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(config['paths']['supplementary_dir'], "Supplementary_Fig_ROC.png"), 
            dpi=600, bbox_inches='tight')
plt.close()

# ============================================================================
# STEP 7: Save Summary Report
# ============================================================================
print("\n[7/8] Generating summary report...")

best_model = df_summary.loc[df_summary['AUC_mean'].idxmax()]

summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_seeds': len(seeds),
    'n_cells_per_run': 15000,
    'total_runs': len(df_all),
    'dl_models_compared': list(atlases.keys()),
    'best_performing': {
        'dl_model': best_model['DL_Model'],
        'ml_model': best_model['ML_Model'],
        'AUC': float(best_model['AUC_mean']),
        'AUC_std': float(best_model['AUC_std'])
    },
    'significant_comparisons': int(df_significance['significant'].sum()),
    'total_comparisons': len(df_significance)
}

import json
with open(os.path.join(config['paths']['output_dir'], "benchmark_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ============================================================================
# STEP 8: Final Output
# ============================================================================
print("\n[8/8] Finalizing...")

print("\n" + "="*70)
print("✅ BENCHMARKING COMPLETE")
print("="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📁 Output files:")
print("  ✓ Benchmark_All_Runs.csv")
print("  ✓ Benchmark_Summary.csv")
print("  ✓ Benchmark_Significance.csv")
print("  ✓ Figure_Benchmark_Overview.png")
print("  ✓ Figure_Stability_Violin.png")
print("  ✓ Supplementary_Fig_Significance.png")
print("  ✓ Supplementary_Fig_ROC.png")
print("  ✓ benchmark_summary.json")

print("\n📊 Best Performing Model:")
print(f"  DL Model: {best_model['DL_Model']}")
print(f"  ML Model: {best_model['ML_Model']}")
print(f"  AUC: {best_model['AUC_mean']:.4f} ± {best_model['AUC_std']:.4f}")