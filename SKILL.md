---
name: stanobj
description: >
  Use when the user asks to "standardize single-cell data", "convert to h5ad",
  "convert Seurat to h5ad", "convert RDS to h5ad", "standardize scRNA-seq",
  "convert 10x to h5ad", "convert CSV/TSV expression matrix", "convert MTX files",
  "read Seurat object", "convert SingleCellExperiment", "harmonize single-cell metadata",
  or mentions converting any single-cell format (rds, mtx, csv, tsv, h5, hdf5, loom)
  into canonical h5ad. Do NOT use for reading/inspecting existing h5ad files
  (that is scrna-reader), gene ID conversion (that is stangene), or biological
  analysis (PCA, clustering, etc).
version: 1.0.0
allowed-tools: [Bash, Read, Glob, Grep]
---

# stanobj — Standardize Single-Cell Data to Canonical h5ad

Convert heterogeneous single-cell transcriptomics datasets into canonical,
analysis-ready `.h5ad` files with full provenance tracking.

## Supported Formats

| Format | Extension | Reader |
|--------|-----------|--------|
| Matrix Market / 10x triplet | `.mtx`, directory | mtx_reader |
| CSV expression matrix | `.csv`, `.csv.gz` | csv_reader |
| TSV expression matrix | `.tsv`, `.tsv.gz` | csv_reader |
| 10x Genomics HDF5 | `.h5` | h5_reader |
| Generic HDF5 | `.h5`, `.hdf5` | h5_reader |
| Existing h5ad | `.h5ad` | h5ad_reader |
| Loom | `.loom` | loom_reader |
| Seurat RDS | `.rds` | seurat_reader (R subprocess) |
| SingleCellExperiment RDS | `.rds` | sce_reader (R subprocess) |

Archives supported: `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar`, `.zip`
Compression: `.gz`, `.bz2` (transparent for most formats)

## Workflow

1. Identify the input file/directory from the user's request
2. Ask the user for an output path (required: `-o <path.h5ad>`)
3. Run the orchestrator:

```bash
python ~/.claude/skills/stanobj/scripts/stanobj.py <input> -o <output.h5ad>
```

4. Check the exit code:
   - **Exit 0**: Success. Read the audit log and present a summary to the user.
   - **Exit 10**: Decision needed. Parse the JSON from stdout, present options to the user (or decide if obvious), then re-invoke with `--decision <type>=<choice>`.
   - **Exit 1**: Fatal error. Read stderr and report to user.

5. After success, optionally suggest running `scrna-reader` to verify the output.

## Decision Protocol

When the orchestrator cannot proceed autonomously, it exits with code 10 and
prints a JSON decision request to stdout:

```json
{
  "status": "decision_needed",
  "decision_type": "assay_selection",
  "context": "Seurat object has 3 assays: RNA, SCT, ADT",
  "options": ["RNA", "SCT", "ADT"],
  "recommendation": "RNA",
  "reason": "RNA has the most features and contains raw counts"
}
```

Re-invoke with: `--decision assay_selection=RNA`

Multiple decisions can be supplied: `--decision assay_selection=RNA --decision matrix_type=counts`

### Known Decision Points

| Type | Trigger | Options |
|------|---------|---------|
| `format_detection` | Unrecognized file format | format names |
| `assay_selection` | Multiple assays in Seurat/SCE | assay names |
| `matrix_orientation` | Ambiguous CSV/TSV layout | `cells_x_genes`, `genes_x_cells` |
| `matrix_type` | Low-confidence classification | `counts`, `normalized`, `log1p`, `scaled`, `unknown` |
| `layer_selection` | Multiple candidate layers | layer names |
| `modality_filter` | Mixed RNA + non-RNA features | `rna_only`, `keep_all` |

## Outputs

Three files per conversion:

```
<output>.h5ad               # Canonical AnnData
<output_stem>_report.json   # Machine-readable report
<output_stem>_audit.log     # Human-readable audit log
```

## Canonical Schema

See `references/canonical-schema.md` for the full target schema.

Key conventions:
- `adata.X` = main analysis matrix
- `adata.layers["counts"]` = raw counts when available
- `adata.obs["cell_id"]` = original cell barcode
- `adata.obs["dataset"]` = dataset name
- `adata.var["gene_symbol"]` = gene symbols
- `adata.var["gene_id"]` = Ensembl IDs if available
- `adata.uns["stanobj"]` = conversion provenance

## Scope Boundaries

- This skill does NOT convert gene IDs — that is `stangene`
- This skill does NOT inspect/compare/merge existing h5ad — that is `scrna-reader`
- This skill does NOT perform biological analysis
