#!/usr/bin/env python3
# ============================================================================
# 05_pathway_simple_manual.py - Manual ORA without external APIs
# Uses pre-downloaded pathway gene sets
# ============================================================================
import pandas as pd
import numpy as np
import os
from scipy.stats import fisher_exact
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== MANUAL PATHWAY ENRICHMENT (No API) ===")

# Load gene weights
weights = pd.read_csv("results/LinearSCVI_Gene_Weights.csv", index_col=0)
z12 = weights['Z_12'].abs().sort_values(ascending=False)
top_genes = z12.head(500).index.tolist()
background_genes = weights.index.tolist()

print(f"  Top genes: {len(top_genes)}")
print(f"  Background: {len(background_genes)}")

# Pre-downloaded pathway database (simplified KEGG-like)
# In practice, download from MSigDB or KEGG beforehand
pathway_db = {
    'Cell Cycle': ['CCNA2', 'CCNB1', 'CCNE1', 'CDK1', 'CDK2', 'E2F1', 'RB1', 'TP53'],
    'Apoptosis': ['BAX', 'BCL2', 'CASP3', 'CASP8', 'FAS', 'TP53'],
    'PI3K-Akt Signaling': ['PIK3CA', 'PIK3R1', 'AKT1', 'AKT2', 'MTOR', 'PTEN'],
    'ECM-Receptor Interaction': ['COL1A1', 'COL4A1', 'FN1', 'ITGA5', 'ITGB1', 'LAMA2'],
    'DNA Repair': ['BRCA1', 'BRCA2', 'RAD51', 'ATM', 'ATR', 'CHEK1'],
    'Immune Response': ['CD3D', 'CD4', 'CD8A', 'IFNG', 'IL6', 'TNF'],
    'Metabolic Pathways': ['HK2', 'LDHA', 'PKM', 'PDHA1', 'SDHA'],
    'EMT': ['CDH1', 'CDH2', 'VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1']
}

# Fisher's exact test for each pathway
results = []
for pathway, pathway_genes in pathway_db.items():
    # 2x2 contingency table
    a = len(set(top_genes) & set(pathway_genes))  # In top genes AND pathway
    b = len(set(top_genes) - set(pathway_genes))  # In top genes NOT in pathway
    c = len(set(background_genes) & set(pathway_genes)) - a  # In pathway NOT in top
    d = len(set(background_genes)) - a - b - c  # Neither
    
    if a + b + c + d == 0 or c + d == 0:
        continue
        
    odds_ratio, p_val = fisher_exact([[a, b], [c, d]], alternative='greater')
    
    results.append({
        'pathway': pathway,
        'overlap': a,
        'pathway_size': a + c,
        'odds_ratio': odds_ratio,
        'p_value': p_val,
        'p_adjusted': p_val * len(pathway_db)  # Bonferroni
    })

df_results = pd.DataFrame(results).sort_values('p_adjusted')
df_results.to_csv("results/Enrichment_Manual_ORA.csv", index=False)

print(f"\n  Found {len(df_results)} pathways tested")
print(f"  Significant (p<0.05): {(df_results['p_adjusted'] < 0.05).sum()}")

# Save top results
if len(df_results) > 0:
    print("\n  Top 5 pathways:")
    print(df_results.head(5)[['pathway', 'overlap', 'p_adjusted']].to_string(index=False))

print("\n✅ Manual enrichment complete!")