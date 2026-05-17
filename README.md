# Deep Learning Latent Representations for Predicting Platinum Resistance in Ovarian Cancer

Code and analysis pipeline accompanying *Lyu et al.* (manuscript under review).

## What this repository contains

Three deep learning architectures (LinearSCVI, standard Autoencoder, Variational Autoencoder) trained on a harmonized single-cell RNA-seq atlas of high-grade serous ovarian carcinoma (HGSOC), benchmarked across multiple classifiers and validation strategies, with downstream pathway, survival, and clinical-utility analyses.

## Data

Two public datasets from the Gene Expression Omnibus:

* **GSE266577** — Launonen et al. 2024, *Cancer Cell*
* **GSE165897** — Zhang et al. 2022, *Science Advances*

We do not redistribute the raw data. Download from GEO directly. Cell-level metadata used in this study (patient ID mapping, PFI category, treatment phase) is in `metadata/`.

## Pipeline order

```
00\_convert\_gene\_ids.py              # Ensembl → HGNC symbols
01\_preprocess\_and\_qc.py             # QC, HVG selection (3,000 genes)
02a\_train\_linearscvi\_SAVEFIX.py     # LinearSCVI training
02b\_train\_autoencoder\_SAVEFIX.py    # Standard autoencoder training
02c\_train\_vae\_WORKING.py            # VAE training
03\_compare\_all\_models.py            # Quick three-way comparison
04\_benchmarking\_stability\_final.py  # Random-split benchmark (10 seeds × 3 classifiers)
04b\_patient\_aware\_benchmark.py      # GroupKFold + LOPO  (PRIMARY result)
04c\_cross\_dataset\_validation.py     # Train on one GSE, test on the other
05\_pathway\_enrichment\_FINAL.py      # MSigDB Hallmark over-representation
06\_survival\_analysis\_FIXED.py       # Patient-level Kaplan–Meier
07\_comprehensive\_visualization\_FIXED.py  # Main figures
08\_clinical\_utility.py              # Decision curve analysis + calibration
10\_negative\_controls.py             # Patient-shuffled permutation test
```

## Reproducibility

```bash
# Python 3.10
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt

# Run end to end (after placing GEO matrices in data/)
bash run\_all.sh
```

Hardware: pipelines ran on a single Linux node, 32 GB RAM, no GPU required (CPU completes the full pipeline in \~6 hours).

## Validation strategies — read before interpreting AUCs

This work uses three cross-validation strategies. Numbers are **not interchangeable** and the paper reports all three for transparency:

|Strategy|Script|What it measures|
|-|-|-|
|Random cell split (10 seeds)|`04\_benchmarking\_stability\_final.py`|Upper bound. Cells from the same patient appear in train and test, so patient identity contributes.|
|GroupKFold by patient (5 folds)|`04b\_patient\_aware\_benchmark.py`|Cell-level performance with **no patient leakage**. The honest cell-level estimate.|
|Leave-one-patient-out|`04b\_patient\_aware\_benchmark.py`|Patient-level performance. The clinically relevant number.|
|Cross-dataset (GSE→GSE)|`04c\_cross\_dataset\_validation.py`|Held-out study generalization. The most stringent test.|

## Contact

Issues and questions: open a GitHub issue or email the corresponding authors.

