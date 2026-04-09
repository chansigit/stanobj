#!/usr/bin/env Rscript
# seurat_reader.R -- Export a Seurat RDS object to 10x-style flat files.
#
# Usage:
#   Rscript seurat_reader.R <input.rds> <output_dir> [--assay <name>]
#
# Phase 1 (inventory, no --assay):
#   Writes JSON to stdout describing available assays.
#   Exit 0  if single assay (auto-selected), or
#   Exit 10 if user must choose among multiple assays.
#   Exit 1  on error (e.g. not a Seurat object).
#
# Phase 2 (export, --assay given):
#   Exports matrix.mtx, barcodes.tsv, features.tsv, metadata.csv,
#   and any dimensionality-reduction CSVs.
#   Exit 0 on success, 1 on error.

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
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
  emit_error("Usage: Rscript seurat_reader.R <input.rds> <output_dir> [--assay <name>]")
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

if (!inherits(obj, "Seurat")) {
  emit_error(
    paste("Expected a Seurat object but got:", detected_class),
    extra = list(detected_class = detected_class)
  )
}

# ---------------------------------------------------------------------------
# Helper: extract a matrix from an assay (v5 Assay5 or legacy Assay)
# ---------------------------------------------------------------------------

get_matrix_from_assay <- function(obj, assay_name) {
  assay_obj <- obj[[assay_name]]
  slot_used <- NULL
  mat <- NULL

  if (inherits(assay_obj, "Assay5")) {
    # Seurat v5 -- use Layers / LayerData
    layers <- Layers(assay_obj)
    if ("counts" %in% layers) {
      mat <- LayerData(assay_obj, layer = "counts")
      slot_used <- "counts"
    } else if ("data" %in% layers) {
      mat <- LayerData(assay_obj, layer = "data")
      slot_used <- "data"
    } else if (length(layers) > 0) {
      mat <- LayerData(assay_obj, layer = layers[1])
      slot_used <- layers[1]
    }
  } else {
    # Legacy Assay -- use GetAssayData with slot argument
    counts_mat <- tryCatch(GetAssayData(obj, assay = assay_name, slot = "counts"),
                           error = function(e) NULL)
    if (!is.null(counts_mat) && prod(dim(counts_mat)) > 0) {
      mat <- counts_mat
      slot_used <- "counts"
    } else {
      data_mat <- tryCatch(GetAssayData(obj, assay = assay_name, slot = "data"),
                           error = function(e) NULL)
      if (!is.null(data_mat) && prod(dim(data_mat)) > 0) {
        mat <- data_mat
        slot_used <- "data"
      }
    }
  }
  list(mat = mat, slot_used = slot_used)
}

# ---------------------------------------------------------------------------
# Inventory: enumerate assays
# ---------------------------------------------------------------------------

assay_names  <- Assays(obj)
default_assay <- DefaultAssay(obj)

assay_info <- lapply(assay_names, function(a) {
  assay_obj <- obj[[a]]
  info <- list(name = a, n_features = nrow(assay_obj))
  if (inherits(assay_obj, "Assay5")) {
    info$type <- "Assay5"
    info$layers <- Layers(assay_obj)
  } else {
    info$type <- "Assay"
    slots_avail <- c()
    for (s in c("counts", "data", "scale.data")) {
      m <- tryCatch(GetAssayData(obj, assay = a, slot = s),
                    error = function(e) NULL)
      if (!is.null(m) && prod(dim(m)) > 0) slots_avail <- c(slots_avail, s)
    }
    info$slots <- slots_avail
  }
  info
})

# ---------------------------------------------------------------------------
# Phase 1: no --assay flag -> inventory / auto-select
# ---------------------------------------------------------------------------

if (is.null(assay_flag)) {
  if (length(assay_names) == 1) {
    # Auto-select the only assay and fall through to export
    assay_flag <- assay_names[1]
  } else {
    # Multiple assays -- ask the user
    emit_json(list(
      status          = "decision_needed",
      decision_type   = "assay_selection",
      options         = assay_names,
      recommendation  = default_assay,
      assay_info      = assay_info,
      n_cells         = ncol(obj)
    ))
    quit(status = 10, save = "no")
  }
}

# ---------------------------------------------------------------------------
# Phase 2: export with chosen assay
# ---------------------------------------------------------------------------

if (!(assay_flag %in% assay_names)) {
  emit_error(paste("Assay not found:", assay_flag,
                   "-- available:", paste(assay_names, collapse = ", ")))
}

res <- get_matrix_from_assay(obj, assay_flag)
mat       <- res$mat
slot_used <- res$slot_used

if (is.null(mat)) {
  emit_error(paste("Could not extract matrix from assay:", assay_flag))
}

# Ensure sparse (dgCMatrix)
if (!inherits(mat, "dgCMatrix")) {
  mat <- as(mat, "dgCMatrix")
}

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Matrix Market (genes x cells)
writeMM(mat, file.path(output_dir, "matrix.mtx"))

# 2. Barcodes
writeLines(colnames(mat), file.path(output_dir, "barcodes.tsv"))

# 3. Features (3 columns: id, symbol, type)
features <- data.frame(
  id     = rownames(mat),
  symbol = rownames(mat),
  type   = "Gene Expression",
  stringsAsFactors = FALSE
)
write.table(features, file.path(output_dir, "features.tsv"),
            sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)

# 4. Cell metadata
meta <- obj@meta.data
write.csv(meta, file.path(output_dir, "metadata.csv"), row.names = TRUE)

# 5. Dimensionality reductions
reductions_exported <- character(0)
red_names <- Reductions(obj)
for (rname in red_names) {
  emb <- tryCatch(Embeddings(obj, reduction = rname), error = function(e) NULL)
  if (!is.null(emb)) {
    fname <- paste0("reduction_", rname, ".csv")
    write.csv(as.data.frame(emb), file.path(output_dir, fname), row.names = TRUE)
    reductions_exported <- c(reductions_exported, rname)
  }
}

# 6. Emit success JSON
emit_json(list(
  status              = "success",
  assay               = assay_flag,
  slot_used           = slot_used,
  n_cells             = ncol(mat),
  n_features          = nrow(mat),
  reductions_exported = reductions_exported
))

quit(status = 0, save = "no")
