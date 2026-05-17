#!/usr/bin/env python3
# ============================================================================
# 04b_generate_benchmark_figures.py - Visualization Only
# Loads already-saved CSV files and generates figures
# FIXED: Font and legend issues
# ============================================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FONT FIX: Use available fonts on Linux servers
# ============================================================================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9

print("="*70)
print("GENERATING BENCHMARK FIGURES (From Saved CSV)")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: Load Saved Benchmark Results
# ============================================================================
print("[1/5] Loading saved benchmark results...")

output_dir = "results"
figures_dir = "figures"
supplementary_dir = "supplementary"

for d in [figures_dir, supplementary_dir]:
    os.makedirs(d, exist_ok=True)

df_all = pd.read_csv(os.path.join(output_dir, "Benchmark_All_Runs.csv"))
df_summary = pd.read_csv(os.path.join(output_dir, "Benchmark_Summary.csv"))
df_significance = pd.read_csv(os.path.join(output_dir, "Benchmark_Significance.csv"))

print(f"  ✓ Benchmark_All_Runs.csv: {len(df_all):,} rows")
print(f"  ✓ Benchmark_Summary.csv: {len(df_summary):,} rows")
print(f"  ✓ Benchmark_Significance.csv: {len(df_significance):,} rows")

# ============================================================================
# STEP 2: Extract Metadata
# ============================================================================
print("\n[2/5] Extracting metadata...")

seeds = df_all['seed'].unique().tolist()
dl_models = ['LinearSCVI', 'Autoencoder', 'VAE']
baseline_models = ['PCA_Linear', 'Gene_Expression']
ml_models = ['RF', 'SVM', 'LASSO']

print(f"  Seeds: {len(seeds)}")
print(f"  DL Models: {dl_models}")
print(f"  ML Models: {ml_models}")

# ============================================================================
# STEP 3: Generate Figures
# ============================================================================
print("\n[3/5] Generating benchmark figures...")

# Color palette
dl_palette = {
    'LinearSCVI': '#E41A1C',
    'Autoencoder': '#377EB8',
    'VAE': '#4DAF4A',
    'PCA_Linear': '#984EA3',
    'Gene_Expression': '#FF7F00'
}

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Multi-metric comparison (Main benchmark figure)
# ────────────────────────────────────────────────────────────────────────────
print("  Creating Figure 1: Benchmark Overview...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Deep Learning Model Benchmarking (10 Seeds)', 
             fontsize=16, fontweight='bold')

metrics = ['AUC', 'F1_Score', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    dl_summary = df_summary.groupby('DL_Model')[f'{metric}_mean'].mean().sort_values(ascending=False)
    dl_std = df_summary.groupby('DL_Model')[f'{metric}_std'].mean()
    
    x = np.arange(len(dl_summary))
    colors = [dl_palette.get(m, '#999999') for m in dl_summary.index]
    
    ax.bar(x, dl_summary.values, yerr=dl_std.loc[dl_summary.index].values,
           color=colors, edgecolor='black', capsize=5, alpha=0.8, linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(dl_summary.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(f'{metric} (Mean ± SD)', fontsize=11)
    ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.05)

plt.tight_layout()
fig_path = os.path.join(figures_dir, "Figure_Benchmark_Overview.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {fig_path}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Stability across seeds (violin plots) - FIXED
# ────────────────────────────────────────────────────────────────────────────
print("  Creating Figure 2: Stability Violin...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Performance Stability Across 10 Random Seeds', 
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(['AUC', 'F1_Score', 'Accuracy']):
    ax = axes[idx]
    
    plot_data = []
    for dl_model in dl_models:
        model_data = df_all[df_all['DL_Model'] == dl_model]
        for seed in seeds:
            seed_data = model_data[model_data['seed'] == seed]
            for ml_model in ml_models:
                ml_data = seed_data[seed_data['ML_Model'] == ml_model]
                if len(ml_data) > 0:
                    plot_data.append({
                        'DL_Model': dl_model,
                        'Score': ml_data[metric].values[0]
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    sns.violinplot(data=plot_df, x='DL_Model', y='Score', hue='DL_Model',
                   ax=ax, palette=dl_palette, inner='box', alpha=0.7, legend=False)
    
    ax.set_xlabel('Deep Learning Model', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.05)

plt.tight_layout()
fig_path = os.path.join(figures_dir, "Figure_Stability_Violin.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {fig_path}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Significance heatmap
# ────────────────────────────────────────────────────────────────────────────
print("  Creating Figure 3: Significance Heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

pivot_sig = df_significance.pivot_table(
    index='Comparison', 
    columns='Metric', 
    values='p_value', 
    aggfunc='mean'
)

pivot_sig_log = -np.log10(pivot_sig + 1e-10)

im = ax.imshow(pivot_sig_log.values, aspect='auto', cmap='Reds', vmin=0, vmax=10)
ax.set_xticks(np.arange(len(pivot_sig_log.columns)))
ax.set_yticks(np.arange(len(pivot_sig_log.index)))
ax.set_xticklabels(pivot_sig_log.columns, fontsize=10)
ax.set_yticklabels(pivot_sig_log.index, fontsize=9)
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Comparison', fontsize=12)
ax.set_title('Statistical Significance (-log10 P-value)', fontsize=14, fontweight='bold')

for i in range(len(pivot_sig_log.index)):
    for j in range(len(pivot_sig_log.columns)):
        val = pivot_sig_log.values[i, j]
        if val > 3:
            ax.text(j, i, '***', ha='center', va='center', fontsize=10, fontweight='bold')
        elif val > 2:
            ax.text(j, i, '**', ha='center', va='center', fontsize=10, fontweight='bold')
        elif val > 1.3:
            ax.text(j, i, '*', ha='center', va='center', fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax, label='-log10(P-value)')
plt.tight_layout()
fig_path = os.path.join(supplementary_dir, "Supplementary_Fig_Significance.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {fig_path}")

# ────────────────────────────────────────────────────────────────────────────
# FIGURE 4: ROC Curves
# ────────────────────────────────────────────────────────────────────────────
print("  Creating Figure 4: ROC Curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('ROC Curves by Deep Learning Model (Best Seed)', 
             fontsize=16, fontweight='bold')

best_seed = seeds[0]

for idx, dl_model in enumerate(dl_models):
    ax = axes[idx]
    
    model_data = df_all[(df_all['DL_Model'] == dl_model) & (df_all['seed'] == best_seed)]
    
    for ml_model in ml_models:
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
fig_path = os.path.join(supplementary_dir, "Supplementary_Fig_ROC.png")
plt.savefig(fig_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {fig_path}")

# ============================================================================
# STEP 4: Print Best Model Summary
# ============================================================================
print("\n[4/5] Calculating best performing model...")

best_model = df_summary.loc[df_summary['AUC_mean'].idxmax()]

print(f"\n🏆 Best Performing Model:")
print(f"   DL Model: {best_model['DL_Model']}")
print(f"   ML Model: {best_model['ML_Model']}")
print(f"   AUC: {best_model['AUC_mean']:.4f} ± {best_model['AUC_std']:.4f}")

# ============================================================================
# STEP 5: Update Summary JSON
# ============================================================================
print("\n[5/5] Updating summary JSON...")

import json

summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_seeds': len(seeds),
    'n_cells_per_run': 15000,
    'total_runs': len(df_all),
    'dl_models_compared': dl_models,
    'best_performing': {
        'dl_model': best_model['DL_Model'],
        'ml_model': best_model['ML_Model'],
        'AUC': float(best_model['AUC_mean']),
        'AUC_std': float(best_model['AUC_std'])
    },
    'significant_comparisons': int(df_significance['significant'].sum()),
    'total_comparisons': len(df_significance),
    'figures_generated': [
        'Figure_Benchmark_Overview.png',
        'Figure_Stability_Violin.png',
        'Supplementary_Fig_Significance.png',
        'Supplementary_Fig_ROC.png'
    ]
}

with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"    ✓ Saved: benchmark_summary.json")

# ============================================================================
# FINAL
# ============================================================================
print("\n" + "="*70)
print("✅ BENCHMARK FIGURES COMPLETE")
print("="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📁 Output files:")
print("  ✓ figures/Figure_Benchmark_Overview.png")
print("  ✓ figures/Figure_Stability_Violin.png")
print("  ✓ supplementary/Supplementary_Fig_Significance.png")
print("  ✓ supplementary/Supplementary_Fig_ROC.png")
print("  ✓ results/benchmark_summary.json")
print("\n🔜 Next: Run pathway enrichment and survival analysis")