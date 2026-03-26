# Decision Points Reference

When the orchestrator encounters ambiguity it cannot resolve confidently, it exits
with code 10 and emits a JSON decision request.

## format_detection
**Trigger:** File format cannot be determined from extension or content.
**Options:** Format names (csv, tsv, mtx, 10x_h5, h5ad, loom, seurat_rds, sce).
**Context:** File path and content summary.

## assay_selection
**Trigger:** Seurat or SCE object has multiple assays.
**Options:** Available assay names.
**Context:** Assay inventory with feature counts and available slots.
**Recommendation:** Default assay (Seurat) or "counts" assay (SCE) if present.

## matrix_orientation
**Trigger:** CSV/TSV with ambiguous row/column labels.
**Options:** `cells_x_genes`, `genes_x_cells`
**Context:** Matrix shape, sample row/column labels.

## matrix_type
**Trigger:** Cannot confidently classify the main matrix.
**Options:** `counts`, `normalized`, `log1p`, `scaled`, `unknown`
**Context:** Evidence metrics (has_negatives, is_integer, max_value, sparsity, row_sums_cv).

## layer_selection
**Trigger:** Multiple candidate layers in h5ad, unclear which is primary.
**Options:** Layer names.
**Context:** Available layers with detected types.

## modality_filter
**Trigger:** Feature types include non-RNA modalities (ADT, CRISPR, Peaks).
**Options:** `rna_only`, `keep_all`
**Context:** Feature type distribution.
