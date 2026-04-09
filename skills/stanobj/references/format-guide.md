# Format-Specific Notes and Gotchas

## CSV / TSV
- Most ambiguous format. Orientation detection uses label pattern matching and shape heuristics.
- First column is assumed to be row identifiers (index_col=0).
- Gene-like patterns: ENSG..., uppercase symbols (TP53), title-case (Trp53).
- Barcode-like patterns: ACGT sequences, prefix-number patterns.
- If neither rows nor columns match known patterns, the orchestrator requests a decision.

## Matrix Market (MTX)
- Convention: genes x cells. Always transposed to cells x genes.
- Look for sibling files: barcodes.tsv[.gz], features.tsv[.gz] or genes.tsv[.gz].
- features.tsv may have 2 columns (id, name) or 3 columns (id, name, feature_type).
- 3rd column indicates modality: Gene Expression, Antibody Capture, CRISPR Guide Capture.

## 10x HDF5 (.h5)
- Contains `matrix/` group with CSC sparse components.
- Internal format is genes x cells. Transposed by reader.
- Features group may include feature_type for multi-modal data.

## Generic HDF5
- No assumed structure. Reader inspects the HDF5 tree.
- If `matrix/` group found: treated as 10x.
- If `X/` or `obs/` found: treated as h5ad-like.
- Otherwise: decision request with tree dump.

## Existing h5ad
- NOT assumed to be canonical.
- Full pipeline runs: validation, standardization, type classification.
- Useful for re-standardizing h5ad files from different pipelines.

## Loom
- Convention: genes x cells (rows = genes, columns = cells).
- Always transposed by reader.
- row_attrs -> var columns, col_attrs -> obs columns.
- "Gene" or "gene" row attribute used for var_names.
- "CellID" or "cellid" col attribute used for obs_names.

## Seurat RDS
- May contain multiple assays: RNA, SCT, ADT, integrated, etc.
- Do NOT assume default assay is RNA.
- Critical slot distinctions: counts = raw, data = log-normalized, scale.data = scaled.
- Exported via R subprocess to temp MTX directory + metadata CSV.
- If RDS contains a SingleCellExperiment instead, automatically redirected.

## SingleCellExperiment RDS
- May contain multiple assays: counts, logcounts, normcounts.
- Exported via R subprocess to temp MTX directory + metadata CSV.
- altExp objects may contain non-RNA modalities (not exported by default).

## R Fallback
- If Rscript is not available, the skill provides manual R commands for conversion.
- Users can run the provided R code, then re-invoke stanobj on the resulting files.
