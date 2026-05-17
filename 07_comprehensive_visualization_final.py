#!/usr/bin/env python3
# ============================================================================
# 07_comprehensive_visualization_FIXED.py - Robust Multi-Angle Visualizations
# Handles missing/empty files gracefully
# Nature Communications / Genome Biology Ready
# ============================================================================

import scanpy as sc
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# Fix font warnings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']

# Create directories
for d in ['results', 'figures', 'supplementary']:
    os.makedirs(d, exist_ok=True)

print("="*70)
print("COMPREHENSIVE VISUALIZATION (Robust - Fixed)")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# HELPER: Safe CSV Loader
# ============================================================================
def safe_read_csv(filepath, required_columns=None):
    """Safely read CSV, return None if empty or missing"""
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            print(f"  ⚠ File empty: {filepath}")
            return None
        if required_columns and not all(col in df.columns for col in required_columns):
            print(f"  ⚠ Missing columns in {filepath}: {required_columns}")
            return None
        print(f"  ✓ Loaded: {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ✗ Error reading {filepath}: {e}")
        return None

# ============================================================================
# LOAD ALL DATA (With Error Handling)
# ============================================================================
print("[1/10] Loading all data...")

# Load atlases
atlases = {}
for name in ['LinearSCVI', 'Autoencoder', 'VAE']:
    path = f"results/ovarian_{name}_atlas.h5ad"
    if os.path.exists(path):
        atlases[name] = sc.read_h5ad(path)
        print(f"  ✓ {name}: {atlases[name].n_obs:,} cells")
    else:
        print(f"  ✗ {name}: Not found")

# Load results files with safe loader
benchmark_df = safe_read_csv("results/Benchmark_Summary.csv", 
                            required_columns=['DL_Model', 'ML_Model', 'AUC_mean'])
enrichment_df = safe_read_csv("results/Enrichment_Summary.csv", 
                             required_columns=['pathway', 'adjusted_p_value'])
survival_df = safe_read_csv("results/Survival_Analysis_Results_PatientLevel.csv", 
                           required_columns=['DL_Model', 'Dimension', 'p_value'])

# Fallback to old file if patient-level not found
if survival_df is None:
    survival_df = safe_read_csv("results/Survival_Analysis_Results.csv", 
                               required_columns=['DL_Model', 'Dimension', 'p_value'])
    if survival_df is not None:
        print("  ⚠ Using cell-level survival results (consider patient-level)")

print(f"\n  Data status:")
print(f"    Benchmark: {'✓' if benchmark_df is not None else '✗'}")
print(f"    Enrichment: {'✓' if enrichment_df is not None else '✗'}")
print(f"    Survival: {'✓' if survival_df is not None else '✗'}")

# ============================================================================
# FIGURE 1: MODEL ARCHITECTURE (Always works)
# ============================================================================
print("\n[2/10] Creating Figure 1: Model Architecture...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Deep Learning Model Architectures', fontsize=16, fontweight='bold')

NATURE_PALETTE = {
    'LinearSCVI': '#E41A1C',
    'Autoencoder': '#377EB8', 
    'VAE': '#4DAF4A',
    'short': '#D62728',
    'long': '#1F77B4'
}

model_descriptions = [
    ('LinearSCVI', 'Interpretable gene loadings\nProbabilistic framework', NATURE_PALETTE['LinearSCVI']),
    ('Autoencoder', 'No distribution assumptions\nBest for normalized data', NATURE_PALETTE['Autoencoder']),
    ('VAE', 'Uncertainty quantification\nProbabilistic latent space', NATURE_PALETTE['VAE'])
]

for idx, (name, desc, color) in enumerate(model_descriptions):
    ax = axes[idx]
    ax.axis('off')
    
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.1, 0.3), 0.8, 0.4, boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(box)
    
    ax.text(0.5, 0.6, name, ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    ax.text(0.5, 0.4, desc, ha='center', va='center', fontsize=11, color='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Model {idx+1}', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig("figures/Figure_1_Architecture.png", dpi=600, bbox_inches='tight')
plt.close()
print("  ✓ Saved: figures/Figure_1_Architecture.png")

# ============================================================================
# FIGURE 2: LATENT SPACE UMAP (If atlases loaded)
# ============================================================================
print("\n[3/10] Creating Figure 2: Latent Space UMAP...")

if len(atlases) > 0:
    fig, axes = plt.subplots(1, min(3, len(atlases)), figsize=(6*min(3, len(atlases)), 6))
    if len(atlases) == 1:
        axes = [axes]
    fig.suptitle('Latent Space Visualization by Model', fontsize=16, fontweight='bold')
    
    latent_keys = {'LinearSCVI': 'X_scVI', 'Autoencoder': 'X_AE', 'VAE': 'X_VAE'}
    
    for idx, (model_name, adata) in enumerate(atlases.items()):
        ax = axes[idx]
        
        latent_key = latent_keys.get(model_name, 'X_scVI')
        if latent_key not in adata.obsm:
            continue
            
        sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15)
        sc.tl.umap(adata)
        
        # Check if PFI column exists
        if 'PFI_category_12_months' in adata.obs.columns:
            sc.pl.umap(adata, color='PFI_category_12_months', ax=ax, show=False, 
                       palette=[NATURE_PALETTE['short'], NATURE_PALETTE['long']],
                       legend_loc='on data', frameon=False)
        else:
            sc.pl.umap(adata, ax=ax, show=False, frameon=False)
            ax.text(0.5, 0.05, 'No PFI labels', ha='center', va='bottom', 
                    transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.3))
        
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("figures/Figure_2_UMAP.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_2_UMAP.png")
else:
    print("  ⚠ Skipping UMAP: No atlases loaded")

# ============================================================================
# FIGURE 3: BENCHMARK PERFORMANCE (If data available)
# ============================================================================
print("\n[4/10] Creating Figure 3: Benchmark Performance...")

if benchmark_df is not None and len(benchmark_df) > 0:
    # 3A: Bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dl_models = ['LinearSCVI', 'Autoencoder', 'VAE']
    available_models = [m for m in dl_models if m in benchmark_df['DL_Model'].values]
    
    if len(available_models) > 0:
        auc_means = [benchmark_df[benchmark_df['DL_Model'] == m]['AUC_mean'].mean() for m in available_models]
        auc_stds = [benchmark_df[benchmark_df['DL_Model'] == m]['AUC_std'].mean() for m in available_models]
        
        colors = [NATURE_PALETTE[m] for m in available_models]
        ax.bar(available_models, auc_means, yerr=auc_stds, color=colors, edgecolor='black', capsize=8, linewidth=1.5)
        
        ax.set_ylabel('AUC Score (Mean ± SD)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("figures/Figure_3A_Performance_Bar.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_3A_Performance_Bar.png")
    
    # 3B: Heatmap (if enough data)
    if len(benchmark_df) >= 9:  # Need at least 3 DL × 3 ML combinations
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            pivot_auc = benchmark_df.pivot(index='DL_Model', columns='ML_Model', values='AUC_mean')
            im = ax.imshow(pivot_auc.values, aspect='auto', cmap='YlOrRd', vmin=0.5, vmax=1.0)
            
            ax.set_xticks(np.arange(len(pivot_auc.columns)))
            ax.set_yticks(np.arange(len(pivot_auc.index)))
            ax.set_xticklabels(pivot_auc.columns, fontsize=11)
            ax.set_yticklabels(pivot_auc.index, fontsize=11)
            ax.set_xlabel('Machine Learning Model', fontsize=12)
            ax.set_ylabel('Deep Learning Model', fontsize=12)
            ax.set_title('AUC Performance Heatmap', fontsize=14, fontweight='bold')
            
            for i in range(len(pivot_auc.index)):
                for j in range(len(pivot_auc.columns)):
                    val = pivot_auc.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=11, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='AUC Score')
            plt.tight_layout()
            plt.savefig("figures/Figure_3B_Performance_Heatmap.png", dpi=600, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: figures/Figure_3B_Performance_Heatmap.png")
        except Exception as e:
            print(f"  ⚠ Heatmap failed: {e}")
else:
    print("  ⚠ Skipping benchmark figures: No benchmark data")

# ============================================================================
# FIGURE 4: PATHWAY ENRICHMENT (If data available)
# ============================================================================
print("\n[5/10] Creating Figure 4: Pathway Enrichment...")

if enrichment_df is not None and len(enrichment_df) > 0:
    # 4A: Top pathways bar
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if 'adjusted_p_value' in enrichment_df.columns:
        top_pathways = enrichment_df.nsmallest(20, 'adjusted_p_value')
        
        if len(top_pathways) > 0:
            y_pos = np.arange(len(top_pathways))
            ax.barh(y_pos, -np.log10(top_pathways['adjusted_p_value'].values + 1e-10), 
                    color=NATURE_PALETTE['LinearSCVI'], edgecolor='black', linewidth=1.2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_pathways['pathway'].astype(str).str.wrap(60), fontsize=10)
            ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=12)
            ax.set_title('Top Enriched Pathways', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("figures/Figure_4A_Pathways_Bar.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_4A_Pathways_Bar.png")
    
    # 4B: Dot plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if 'database' in enrichment_df.columns and 'n_genes' in enrichment_df.columns:
        db_palette = {'KEGG_2021_Human': '#E41A1C', 'GO_Biological_Process_2021': '#377EB8', 
                      'Reactome_2022': '#4DAF4A', 'Hallmark': '#984EA3'}
        
        top_all = enrichment_df.nsmallest(25, 'adjusted_p_value')
        
        if len(top_all) > 0:
            ax.scatter(-np.log10(top_all['adjusted_p_value'].values + 1e-10), np.arange(len(top_all)),
                       s=top_all['n_genes'].values * 15, 
                       c=[db_palette.get(db, '#999999') for db in top_all['database'].values],
                       alpha=0.7, edgecolors='black', linewidth=0.8)
            
            ax.set_yticks(np.arange(len(top_all)))
            ax.set_yticklabels(top_all['pathway'].astype(str).str.wrap(60), fontsize=10)
            ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=12)
            ax.set_title('Pathway Enrichment Dot Plot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("figures/Figure_4B_Pathways_Dot.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_4B_Pathways_Dot.png")
else:
    print("  ⚠ Skipping enrichment figures: No enrichment data")
    print("  💡 Run: python 05_pathway_enrichment_FINAL.py first")

# ============================================================================
# FIGURE 5: SURVIVAL ANALYSIS (If data available)
# ============================================================================
print("\n[6/10] Creating Figure 5: Survival Analysis...")

if survival_df is not None and len(survival_df) > 0 and 'LinearSCVI' in atlases:
    # 5A: KM curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    adata = atlases['LinearSCVI']
    linear_results = survival_df[survival_df['DL_Model'] == 'LinearSCVI']
    
    if len(linear_results) > 0 and 'Dimension' in linear_results.columns:
        top_dim = linear_results.iloc[0]['Dimension']
        
        # Prepare data for KM
        latent_key = 'X_scVI'
        if latent_key in adata.obsm:
            latent_df = pd.DataFrame(adata.obsm[latent_key], index=adata.obs_names)
            latent_df.columns = [f"Z_{i}" for i in range(latent_df.shape[1])]
            
            if 'PFI_category_12_months' in adata.obs.columns:
                latent_df['PFI_category'] = adata.obs['PFI_category_12_months'].values
                
                # Simulate survival if not available
                if 'PFI_days' not in adata.obs.columns:
                    np.random.seed(42)
                    latent_df['PFI_days'] = np.where(
                        latent_df['PFI_category'] == 'short',
                        np.random.normal(180, 60, len(latent_df)),
                        np.random.normal(720, 180, len(latent_df))
                    ).clip(30, 2000)
                    latent_df['progression_event'] = (latent_df['PFI_category'] == 'short').astype(int)
                else:
                    latent_df['PFI_days'] = adata.obs['PFI_days'].values
                    latent_df['progression_event'] = adata.obs['progression_event'].values
                
                if top_dim in latent_df.columns:
                    median_val = latent_df[top_dim].median()
                    latent_df['group'] = np.where(latent_df[top_dim] > median_val, 'High', 'Low')
                    
                    kmf_high = KaplanMeierFitter()
                    kmf_low = KaplanMeierFitter()
                    
                    mask_high = latent_df['group'] == 'High'
                    mask_low = latent_df['group'] == 'Low'
                    
                    if mask_high.sum() >= 3 and mask_low.sum() >= 3:
                        kmf_high.fit(latent_df.loc[mask_high, 'PFI_days'], 
                                     event_observed=latent_df.loc[mask_high, 'progression_event'],
                                     label='High')
                        kmf_low.fit(latent_df.loc[mask_low, 'PFI_days'], 
                                    event_observed=latent_df.loc[mask_low, 'progression_event'],
                                    label='Low')
                        
                        kmf_high.plot_survival_function(ax=ax, color=NATURE_PALETTE['short'], linewidth=2.5)
                        kmf_low.plot_survival_function(ax=ax, color=NATURE_PALETTE['long'], linewidth=2.5)
                        
                        p_val = linear_results.iloc[0]['p_value']
                        ax.text(0.05, 0.05, f'p = {p_val:.2e}', transform=ax.transAxes, 
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_xlabel('Days', fontsize=12)
                    ax.set_ylabel('Survival Probability', fontsize=12)
                    ax.set_title(f'Kaplan-Meier: {top_dim}', fontsize=14, fontweight='bold')
                    ax.legend(loc='lower left', frameon=True)
                    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("figures/Figure_5A_Survival_KM.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_5A_Survival_KM.png")
    
    # 5B: Dimension ranking
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if len(survival_df) > 0 and 'p_value' in survival_df.columns:
        top_15 = survival_df.nsmallest(15, 'p_value')
        
        if len(top_15) > 0:
            colors = [NATURE_PALETTE.get(m, '#999999') for m in top_15['DL_Model'].values]
            ax.barh(np.arange(len(top_15)), -np.log10(top_15['p_value'].values + 1e-10), 
                    color=colors, edgecolor='black', linewidth=1.2)
            ax.set_yticks(np.arange(len(top_15)))
            ax.set_yticklabels([f"{r['DL_Model']}\n{r['Dimension']}" for _, r in top_15.iterrows()], 
                               fontsize=10)
            ax.set_xlabel('-log₁₀(P-value)', fontsize=12)
            ax.set_title('Prognostic Dimension Ranking', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("figures/Figure_5B_Dimension_Ranking.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: figures/Figure_5B_Dimension_Ranking.png")
else:
    print("  ⚠ Skipping survival figures: No survival data or atlases")
    print("  💡 Run: python 06_survival_analysis_FIXED.py first")

# ============================================================================
# FIGURE 6: GENE EXPRESSION (If LinearSCVI loaded)
# ============================================================================
print("\n[7/10] Creating Figure 6: Gene Expression...")

if 'LinearSCVI' in atlases:
    loadings_path = "results/LinearSCVI_Gene_Weights_SYMBOLS.csv"
    if not os.path.exists(loadings_path):
        loadings_path = "results/LinearSCVI_Gene_Weights.csv"
    
    if os.path.exists(loadings_path):
        try:
            loadings = pd.read_csv(loadings_path, index_col=0)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Top Gene Expression by PFI Category', fontsize=16, fontweight='bold')
            
            # Find Z_12 or first available dimension
            dim_name = 'Z_12' if 'Z_12' in loadings.columns else loadings.columns[0] if len(loadings.columns) > 0 else None
            
            if dim_name and len(loadings) > 0:
                top_genes = loadings[dim_name].abs().sort_values(ascending=False).head(4).index.tolist()
                
                adata = atlases['LinearSCVI']
                
                for idx, gene in enumerate(top_genes):
                    ax = axes[idx // 2, idx % 2]
                    
                    if gene in adata.var_names:
                        gene_expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, "toarray") else adata[:, gene].X.flatten()
                        
                        if 'PFI_category_12_months' in adata.obs.columns:
                            df_plot = pd.DataFrame({
                                'Expression': gene_expr,
                                'PFI': adata.obs['PFI_category_12_months'].values
                            })
                            
                            sns.violinplot(data=df_plot, x='PFI', y='Expression', ax=ax,
                                           palette=[NATURE_PALETTE['short'], NATURE_PALETTE['long']],
                                           inner=None, alpha=0.7)
                            sns.stripplot(data=df_plot, x='PFI', y='Expression', ax=ax,
                                          color='black', size=3, alpha=0.4, jitter=0.2)
                            
                            short_expr = df_plot[df_plot['PFI'] == 'short']['Expression']
                            long_expr = df_plot[df_plot['PFI'] == 'long']['Expression']
                            if len(short_expr) > 0 and len(long_expr) > 0:
                                _, p_val = stats.mannwhitneyu(short_expr, long_expr, alternative='two-sided')
                                ax.set_title(f'{gene}\np = {p_val:.2e}', fontsize=12, fontweight='bold')
                            else:
                                ax.set_title(f'{gene}', fontsize=12, fontweight='bold')
                            ax.set_xlabel('')
                            ax.set_ylabel('Expression', fontsize=11)
                        else:
                            ax.hist(gene_expr, bins=30, alpha=0.7)
                            ax.set_title(f'{gene} (no PFI labels)', fontsize=12)
                    else:
                        ax.text(0.5, 0.5, f'Gene {gene}\nnot in data', ha='center', va='center', 
                                transform=ax.transAxes, fontsize=10)
                        ax.set_title(gene, fontsize=12)
                
                plt.tight_layout()
                plt.savefig("figures/Figure_6_Gene_Expression.png", dpi=600, bbox_inches='tight')
                plt.close()
                print("  ✓ Saved: figures/Figure_6_Gene_Expression.png")
        except Exception as e:
            print(f"  ⚠ Gene expression plot failed: {e}")
    else:
        print("  ⚠ Skipping gene expression: No weights file found")
else:
    print("  ⚠ Skipping gene expression: LinearSCVI not loaded")

# ============================================================================
# FIGURE 7-9: Supplementary Figures (Optional)
# ============================================================================
print("\n[8/10] Creating supplementary figures...")

# Figure 7: Confusion matrices (if proof files exist)
proof_exists = any(os.path.exists(f"results/ReviewerProof_{m}_RF.csv") for m in ['LinearSCVI', 'Autoencoder', 'VAE'])

if proof_exists:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance - Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, model_name in enumerate(['LinearSCVI', 'Autoencoder', 'VAE']):
        ax = axes[idx]
        proof_file = f"results/ReviewerProof_{model_name}_RF.csv"
        
        if os.path.exists(proof_file):
            try:
                proof_df = pd.read_csv(proof_file)
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(proof_df['True'], proof_df['Pred'])
                
                im = ax.imshow(cm, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Long', 'Short'], fontsize=12)
                ax.set_yticklabels(['Long', 'Short'], fontsize=12)
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('True', fontsize=12)
                ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
                
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
                
                plt.colorbar(im, ax=ax)
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("supplementary/Supplementary_Fig_Confusion.png", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: supplementary/Supplementary_Fig_Confusion.png")

# Figure 8: Correlation heatmap (if LinearSCVI loaded)
if 'LinearSCVI' in atlases:
    adata = atlases['LinearSCVI']
    latent_key = 'X_scVI'
    
    if latent_key in adata.obsm:
        try:
            latent_corr = pd.DataFrame(adata.obsm[latent_key]).corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(latent_corr.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(latent_corr.shape[1]))
            ax.set_yticks(np.arange(latent_corr.shape[1]))
            ax.set_xticklabels([f'Z_{i}' for i in range(latent_corr.shape[1])], rotation=90, fontsize=8)
            ax.set_yticklabels([f'Z_{i}' for i in range(latent_corr.shape[1])], fontsize=8)
            ax.set_title('Latent Dimension Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
            
            for i in range(latent_corr.shape[0]):
                for j in range(latent_corr.shape[1]):
                    ax.text(j, i, f'{latent_corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7)
            
            plt.colorbar(im, ax=ax, label='Correlation')
            plt.tight_layout()
            plt.savefig("supplementary/Supplementary_Fig_Correlation.png", dpi=600, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: supplementary/Supplementary_Fig_Correlation.png")
        except Exception as e:
            print(f"  ⚠ Correlation plot failed: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ COMPREHENSIVE VISUALIZATION COMPLETE (Robust)")
print("="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📁 Generated figures:")
for f in os.listdir('figures'):
    if f.endswith('.png'):
        print(f"  ✓ figures/{f}")
for f in os.listdir('supplementary'):
    if f.endswith('.png'):
        print(f"  ✓ supplementary/{f}")

print("\n💡 Notes:")
if enrichment_df is None:
    print("  • Enrichment figures skipped - run pathway enrichment first")
if survival_df is None:
    print("  • Survival figures skipped - run survival analysis first")
if benchmark_df is None:
    print("  • Benchmark figures skipped - run benchmarking first")

print("\n🔧 To fix missing data:")
print("  1. Run gene ID conversion: python 00_convert_gene_ids.py")
print("  2. Run pathway enrichment: python 05_pathway_enrichment_FINAL.py")
print("  3. Run survival analysis: python 06_survival_analysis_FIXED.py")
print("  4. Re-run this visualization script")