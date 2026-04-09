#!/usr/bin/env Rscript
# sce_reader.R -- Export a SingleCellExperiment RDS to 10x-style flat files.
#
# Usage:
#   Rscript sce_reader.R <input.rds> <output_dir> [--assay <name>]
#
# Phase 1 (inventory, no --assay):
#   Auto-selects "counts" assay if available. Otherwise emits
#   decision_needed JSON and exits 10.
#
# Phase 2 (export, --assay given):
#   Exports matrix.mtx, barcodes.tsv, features.tsv, metadata.csv,
#   and dimensionality-reduction CSVs.
#   Exit 0 on success, 1 on error.

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(Matrix)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

emit_json <- function(lst) {
  cat(toJSON(lst, auto_unbox = TRUE, pretty = TRUE), "\n")
}

emit_error <- function(msg, extra = list()) {
  info <- c(list(status = "error", message = msg), extra)
  emit_json(info)
  quit(status = 1, save = "no")
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  emit_error("Usage: Rscript sce_reader.R <input.rds> <output_dir> [--assay <name>]")
}

input_path  <- args[1]
output_dir  <- args[2]
assay_flag  <- NULL

if (length(args) >= 4 && args[3] == "--assay") {
  assay_flag <- args[4]
}

# ---------------------------------------------------------------------------
# Load RDS and verify class
# ---------------------------------------------------------------------------

if (!file.exists(input_path)) {
  emit_error(paste("File not found:", input_path))
}

obj <- tryCatch(readRDS(input_path), error = function(e) {
  emit_error(paste("Failed to read RDS:", conditionMessage(e)))
})

detected_class <- class(obj)[1]

if (!inherits(obj, "SingleCellExperiment")) {
  emit_error(
    paste("Expected a SingleCellExperiment but got:", detected_class),
    extra = list(detected_class = detected_class)
  )
}

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

assay_names     <- assayNames(obj)
red_dim_names   <- reducedDimNames(obj)
alt_exp_names   <- altExpNames(obj)
col_data_cols   <- colnames(colData(obj))
row_data_cols   <- colnames(rowData(obj))

assay_info <- lapply(assay_names, function(a) {
  m <- assay(obj, a)
  list(name = a, n_features = nrow(m), n_cells = ncol(m),
       class = class(m)[1])
})

# ---------------------------------------------------------------------------
# Phase 1: no --assay -> auto-select or ask
# ---------------------------------------------------------------------------

if (is.null(assay_flag)) {
  if ("counts" %in% assay_names) {
    assay_flag <- "counts"
  } else if (length(assay_names) == 1) {
    assay_flag <- assay_names[1]
  } else {
    recommendation <- assay_names[1]
    emit_json(list(
      status         = "decision_needed",
      decision_type  = "assay_selection",
      options        = assay_names,
      recommendation = recommendation,
      assay_info     = assay_info,
      reducedDims    = red_dim_names,
      altExps        = alt_exp_names,
      colData_cols   = col_data_cols,
      rowData_cols   = row_data_cols,
      n_cells        = ncol(obj)
    ))
    quit(status = 10, save = "no")
  }
}

# ---------------------------------------------------------------------------
# Phase 2: export
# ---------------------------------------------------------------------------

if (!(assay_flag %in% assay_names)) {
  emit_error(paste("Assay not found:", assay_flag,
                   "-- available:", paste(assay_names, collapse = ", ")))
}

mat <- assay(obj, assay_flag)

# Ensure sparse
if (!inherits(mat, "dgCMatrix")) {
  mat <- as(mat, "dgCMatrix")
}

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Matrix Market (genes x cells)
writeMM(mat, file.path(output_dir, "matrix.mtx"))

# 2. Barcodes
barcodes <- colnames(mat)
if (is.null(barcodes)) barcodes <- paste0("cell_", seq_len(ncol(mat)))
writeLines(barcodes, file.path(output_dir, "barcodes.tsv"))

# 3. Features
gene_ids <- rownames(mat)
if (is.null(gene_ids)) gene_ids <- paste0("gene_", seq_len(nrow(mat)))
features <- data.frame(
  id     = gene_ids,
  symbol = gene_ids,
  type   = "Gene Expression",
  stringsAsFactors = FALSE
)
write.table(features, file.path(output_dir, "features.tsv"),
            sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)

# 4. Cell metadata (colData)
meta <- as.data.frame(colData(obj))
write.csv(meta, file.path(output_dir, "metadata.csv"), row.names = TRUE)

# 5. Dimensionality reductions
reductions_exported <- character(0)
for (rname in red_dim_names) {
  emb <- tryCatch(as.data.frame(reducedDim(obj, rname)), error = function(e) NULL)
  if (!is.null(emb)) {
    rownames(emb) <- colnames(mat)
    fname <- paste0("reduction_", rname, ".csv")
    write.csv(emb, file.path(output_dir, fname), row.names = TRUE)
    reductions_exported <- c(reductions_exported, rname)
  }
}

# 6. Determine slot_hint from assay name
slot_hint <- assay_flag
if (assay_flag == "counts")     slot_hint <- "counts"
if (assay_flag == "logcounts")  slot_hint <- "logcounts"
if (assay_flag == "normcounts") slot_hint <- "normcounts"

# 7. Emit success JSON
emit_json(list(
  status              = "success",
  assay               = assay_flag,
  slot_hint           = slot_hint,
  n_cells             = ncol(mat),
  n_features          = nrow(mat),
  reductions_exported = reductions_exported,
  altExps             = alt_exp_names,
  colData_cols        = col_data_cols,
  rowData_cols        = row_data_cols
))

quit(status = 0, save = "no")
