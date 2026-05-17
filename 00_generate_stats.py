#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
from datetime import datetime

print("=== GENERATING SUMMARY STATISTICS ===")

# Load QC metrics
qc = pd.read_csv("results/QC_metrics.tsv", sep="\t")
hvg = pd.read_csv("AI_READY_OVARIAN_DATA.tsv", sep="\t")

# Load original data info
original = pd.read_csv("AI_READY_OVARIAN_DATA.tsv", sep="\t", nrows=10)
n_original_cells = len(pd.read_csv("AI_READY_OVARIAN_DATA.tsv", sep="\t", usecols=[0]))
n_original_genes = len(original.columns) - 5

# Load DL input
dl_input = pd.read_csv("results/DL_INPUT_3000_HVG.tsv", sep="\t")

# Patient distribution
patient_dist = qc['patient_ID'].value_counts()
pfi_dist = qc['PFI_category_12_months'].value_counts()

summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'original_dataset': {
        'total_cells': int(n_original_cells),
        'total_genes': int(n_original_genes)
    },
    'after_qc': {
        'cells_retained': int(len(qc)),
        'cells_removed': int(n_original_cells - len(qc)),
        'retention_rate': f"{100*len(qc)/n_original_cells:.1f}%"
    },
    'gene_selection': {
        'hvg_selected': int(len(hvg)),
        'final_features': int(len(dl_input.columns) - 6)
    },
    'patient_statistics': {
        'total_patients': int(qc['patient_ID'].nunique()),
        'cells_per_patient_median': float(qc.groupby('patient_ID').size().median()),
        'cells_per_patient_mean': float(qc.groupby('patient_ID').size().mean())
    },
    'pfi_distribution': {
        'short_PFI': int(pfi_dist.get('short', 0)),
        'long_PFI': int(pfi_dist.get('long', 0)),
        'short_percentage': f"{100*pfi_dist.get('short', 0)/len(qc):.1f}%"
    },
    'treatment_phases': qc['treatment_phase'].value_counts().to_dict()
}

with open("results/Summary_Statistics.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"""
📊 PREPROCESSING SUMMARY
{'='*50}
Original dataset:     {n_original_cells:,} cells × {n_original_genes:,} genes
After QC:             {len(qc):,} cells ({100*len(qc)/n_original_cells:.1f}% retained)
After HVG selection:  {len(hvg)} genes

👥 PATIENT STATISTICS
{'='*50}
Total patients:       {qc['patient_ID'].nunique()}
Median cells/patient: {qc.groupby('patient_ID').size().median():.0f}

📈 PFI DISTRIBUTION
{'='*50}
Short PFI (<12 mo):   {pfi_dist.get('short', 0):,} cells ({100*pfi_dist.get('short', 0)/len(qc):.1f}%)
Long PFI (≥12 mo):    {pfi_dist.get('long', 0):,} cells ({100*pfi_dist.get('long', 0)/len(qc):.1f}%)

✅ Saved: results/Summary_Statistics.json
""")