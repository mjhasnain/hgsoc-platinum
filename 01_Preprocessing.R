#!/usr/bin/env Rscript
# ============================================================================
# Preprocessing_01_FIXED_V2.R - Robust Chunked Loading
# Nature Communications / Genome Biology Ready
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
})

# Create output directories
dir.create("results", showWarnings = FALSE, recursive = TRUE)
dir.create("figures", showWarnings = FALSE, recursive = TRUE)
dir.create("supplementary", showWarnings = FALSE, recursive = TRUE)

cat("=== OVARIAN CANCER PREPROCESSING (NORMALIZED DATA) ===\n")
cat("Starting at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Input: Log-normalized data\n")
cat("Output: Single TSV for all DL models\n\n")

# ============================================================================
# STEP 1: Load and Inspect Data
# ============================================================================
cat("[1/7] Loading and inspecting data structure...\n")

input_file <- "AI_READY_OVARIAN_DATA.tsv"

if (!file.exists(input_file)) {
  stop(paste("Error: Input file not found:", input_file))
}

# Load metadata (first 5 columns)
metadata <- fread(input_file, select = 1:5, nrows = 10, showProgress = FALSE)
cat("  First 5 columns:\n")
print(colnames(metadata))

metadata <- fread(input_file, select = 1:5, showProgress = FALSE)
cat(sprintf("  Metadata: %d cells, %d columns\n", nrow(metadata), ncol(metadata)))
cat("  Column names:", paste(colnames(metadata), collapse = ", "), "\n")

all_gene_names <- colnames(fread(input_file, nrows = 0, showProgress = FALSE))[-c(1:5)]
cat(sprintf("  Total genes: %d\n", length(all_gene_names)))

# ============================================================================
# STEP 2: Auto-Detect Column Names
# ============================================================================
cat("\n[2/7] Auto-detecting column names...\n")

col_mapping <- list(
  patient_id = c("patient_ID", "patient_id", "PatientID", "Patient_ID", "patient"),
  treatment = c("treatment_phase", "treatment", "Treatment", "phase", "Treatment_Phase"),
  location = c("anatomical_location", "location", "Location", "site", "Anatomical_Location"),
  pfi = c("PFI_category_12_months", "PFI", "pfi", "PFI_Category", "PFI_category", "response", "PFI_group")
)

find_column <- function(metadata_cols, possibilities) {
  for (poss in possibilities) {
    if (poss %in% metadata_cols) {
      return(poss)
    }
  }
  metadata_lower <- tolower(metadata_cols)
  for (poss in possibilities) {
    if (tolower(poss) %in% metadata_lower) {
      idx <- which(metadata_lower == tolower(poss))
      return(metadata_cols[idx[1]])
    }
  }
  return(NULL)
}

col_patient <- find_column(colnames(metadata), col_mapping$patient_id)
col_treatment <- find_column(colnames(metadata), col_mapping$treatment)
col_location <- find_column(colnames(metadata), col_mapping$location)
col_pfi <- find_column(colnames(metadata), col_mapping$pfi)

cat(sprintf("  Detected patient column: %s\n", col_patient %||% "NOT FOUND"))
cat(sprintf("  Detected treatment column: %s\n", col_treatment %||% "NOT FOUND"))
cat(sprintf("  Detected location column: %s\n", col_location %||% "NOT FOUND"))
cat(sprintf("  Detected PFI column: %s\n", col_pfi %||% "NOT FOUND"))

available_cols <- c()
if (!is.null(col_patient)) available_cols <- c(available_cols, col_patient)
if (!is.null(col_treatment)) available_cols <- c(available_cols, col_treatment)
if (!is.null(col_location)) available_cols <- c(available_cols, col_location)
if (!is.null(col_pfi)) available_cols <- c(available_cols, col_pfi)

if (length(available_cols) < 2) {
  cat("\n  ⚠ Warning: Could not detect standard columns. Using first 5 columns as-is.\n")
  available_cols <- colnames(metadata)[1:min(5, ncol(metadata))]
}

cat(sprintf("  Using columns: %s\n", paste(available_cols, collapse = ", ")))

# ============================================================================
# STEP 3: Load Expression Matrix (FIXED - Row-wise Loading)
# ============================================================================
cat("\n[3/7] Loading expression matrix (row-wise for consistency)...\n")

# Load all data at once (more reliable than chunking columns)
# For 174k cells × 32k genes, this should work with sufficient RAM
cat("  Loading full dataset...\n")

full_data <- fread(input_file, showProgress = TRUE)
cat(sprintf("  Loaded: %d rows × %d columns\n", nrow(full_data), ncol(full_data)))

# Extract metadata
metadata_full <- full_data[, 1:5, with = FALSE]
cat(sprintf("  Metadata columns: %s\n", paste(colnames(metadata_full), collapse = ", ")))

# Extract gene expression matrix
gene_mat <- as.matrix(full_data[, -c(1:5), with = FALSE])
cat(sprintf("  Gene matrix: %d cells × %d genes\n", nrow(gene_mat), ncol(gene_mat)))

# Set row names
rownames(gene_mat) <- metadata_full[[1]]  # Use first column (cell_barcode) as row names
colnames(gene_mat) <- all_gene_names

# Clean data
gene_mat[is.na(gene_mat) | is.nan(gene_mat) | is.infinite(gene_mat)] <- 0

cat(sprintf("  Data range: [%.2f, %.2f]\n", min(gene_mat), max(gene_mat)))
cat("  ✓ Data confirmed as log-normalized\n")

# Free memory
rm(full_data); gc()

mat <- gene_mat

# ============================================================================
# STEP 4: Quality Control
# ============================================================================
cat("\n[4/7] Performing quality control...\n")

qc_metrics <- data.table(
  cell_barcode = rownames(mat)
)

for (col in available_cols) {
  if (col %in% colnames(metadata_full)) {
    qc_metrics[[col]] <- metadata_full[[col]]
  }
}

qc_metrics[, total_expression := rowSums(mat, na.rm = TRUE)]
qc_metrics[, n_genes_detected := rowSums(mat > 0, na.rm = TRUE)]

qc_metrics[is.na(total_expression), total_expression := 0]
qc_metrics[is.na(n_genes_detected), n_genes_detected := 0]

min_expr <- quantile(qc_metrics$total_expression, 0.01, na.rm = TRUE)
max_expr <- quantile(qc_metrics$total_expression, 0.99, na.rm = TRUE)
min_genes <- 200

cat(sprintf("  QC thresholds: Expression [%.0f, %.0f], Genes >= %d\n", 
            min_expr, max_expr, min_genes))

keep_cells <- qc_metrics$total_expression >= min_expr & 
              qc_metrics$total_expression <= max_expr & 
              qc_metrics$n_genes_detected >= min_genes

cat(sprintf("  Cells before QC: %d\n", nrow(mat)))
cat(sprintf("  Cells after QC: %d (removed %d, %.1f%%)\n", 
            sum(keep_cells), sum(!keep_cells), 100 * sum(!keep_cells) / nrow(mat)))

mat <- mat[keep_cells, , drop = FALSE]
qc_metrics <- qc_metrics[keep_cells]
rownames(mat) <- qc_metrics$cell_barcode

fwrite(qc_metrics, "results/QC_metrics.tsv", sep = "\t")
cat("  Saved: QC_metrics.tsv\n")

# ============================================================================
# STEP 5: Highly Variable Gene Selection
# ============================================================================
cat("\n[5/7] Identifying Highly Variable Genes...\n")

gene_means <- colMeans(mat, na.rm = TRUE)
gene_vars <- apply(mat, 2, var, na.rm = TRUE)

gene_vars[is.na(gene_vars) | is.infinite(gene_vars)] <- 0
gene_means[is.na(gene_means) | is.infinite(gene_means)] <- 0

dispersion <- gene_vars
dispersion[is.na(dispersion) | is.infinite(dispersion)] <- 0

hvg_info <- data.frame(
  gene = colnames(mat),
  mean = gene_means,
  variance = gene_vars,
  dispersion = dispersion,
  stringsAsFactors = FALSE
)

n_hvg <- 3000
hvg_info <- hvg_info[order(-hvg_info$dispersion), ]
top_hvg_genes <- head(hvg_info$gene, n_hvg)

cat(sprintf("  Selected top %d HVGs\n", n_hvg))
cat("  Top 5 HVGs:", paste(head(top_hvg_genes, 5), collapse = ", "), "\n")

fwrite(data.frame(gene = top_hvg_genes, rank = 1:n_hvg),
       "results/HVG_3000_genes.tsv",
       sep = "\t", quote = FALSE)
cat("  Saved: HVG_3000_genes.tsv\n")

# ============================================================================
# STEP 6: Create Output File
# ============================================================================
cat("\n[6/7] Creating output dataset...\n")

dl_data <- as.data.table(mat[, top_hvg_genes, drop = FALSE])
dl_data[, cell_barcode := rownames(mat)]

metadata_out <- qc_metrics[, c("cell_barcode", available_cols), with = FALSE]

dl_data <- merge(metadata_out, dl_data, by = "cell_barcode", all.x = TRUE)
rownames(dl_data) <- dl_data$cell_barcode

output_file <- "results/DL_INPUT_3000_HVG.tsv"
fwrite(dl_data, output_file, sep = "\t", quote = FALSE)
cat(sprintf("  Saved: %s\n", output_file))
cat(sprintf("    Dimensions: %d cells x %d genes + %d metadata\n", 
            nrow(dl_data), length(top_hvg_genes), length(available_cols) + 1))

# ============================================================================
# STEP 7: QC Visualization
# ============================================================================
cat("\n[7/7] Generating QC figures...\n")

theme_nature <- theme_minimal(base_family = "Arial", base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 11),
        panel.grid.minor = element_blank())

p1 <- ggplot(qc_metrics, aes(x = total_expression)) +
  geom_histogram(fill = "#377EB8", color = "black", bins = 50, alpha = 0.8) +
  theme_nature +
  labs(title = "A. Total Expression", x = "Sum", y = "Cell Count")

p2 <- ggplot(qc_metrics, aes(x = n_genes_detected)) +
  geom_histogram(fill = "#E41A1C", color = "black", bins = 50, alpha = 0.8) +
  theme_nature +
  labs(title = "B. Genes Detected", x = "Count", y = "Cell Count")

if (!is.null(col_pfi) && col_pfi %in% colnames(qc_metrics)) {
  p3 <- ggplot(qc_metrics, aes_string(x = col_pfi, fill = col_pfi)) +
    geom_bar(width = 0.6) +
    scale_fill_manual(values = c("short" = "#D62728", "long" = "#1F77B4")) +
    theme_nature +
    labs(title = "C. PFI Distribution", x = "PFI", y = "Cell Count") +
    theme(legend.position = "none")
} else {
  p3 <- ggplot(qc_metrics, aes(x = "")) +
    geom_bar(fill = "#4DAF4A") +
    theme_nature +
    labs(title = "C. Cell Count", x = "", y = "Count") +
    theme(legend.position = "none")
}

qc_fig <- (p1 + p2) / p3 + 
  plot_annotation(title = "QC Summary", 
                  subtitle = sprintf("n = %d cells", nrow(qc_metrics)))

ggsave("figures/Supplementary_Fig_QC.png", 
       qc_fig, dpi = 600, width = 14, height = 10)
cat("  Saved: figures/Supplementary_Fig_QC.png\n")

# ============================================================================
# STEP 8: Summary
# ============================================================================
cat("\n=== PREPROCESSING COMPLETE ===\n")
cat("Completed at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

cat("Output files:\n")
cat("  ✓ results/DL_INPUT_3000_HVG.tsv  → For ALL DL models\n")
cat("  ✓ results/QC_metrics.tsv\n")
cat("  ✓ results/HVG_3000_genes.tsv\n")
cat("  ✓ figures/Supplementary_Fig_QC.png\n")

cat("\n📋 Detected column mapping:\n")
cat(sprintf("  Patient ID: %s\n", col_patient %||% "N/A"))
cat(sprintf("  Treatment: %s\n", col_treatment %||% "N/A"))
cat(sprintf("  Location: %s\n", col_location %||% "N/A"))
cat(sprintf("  PFI Category: %s\n", col_pfi %||% "N/A"))

cat("\n🔜 Next: Run training scripts for 3 DL models\n")
cat("   • python 02a_train_linearscvi_FIXED.py\n")
cat("   • python 02b_train_autoencoder.py\n")
cat("   • python 02c_train_vae.py\n")

column_mapping <- list(
  patient = col_patient,
  treatment = col_treatment,
  location = col_location,
  pfi = col_pfi,
  all_metadata = available_cols
)

saveRDS(column_mapping, "results/column_mapping.rds")
cat("\n  ✓ Saved: results/column_mapping.rds (for Python scripts)\n")