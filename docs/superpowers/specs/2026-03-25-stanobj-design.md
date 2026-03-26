# stanobj — Standardize Heterogeneous Single-Cell Transcriptomics into Canonical h5ad

**Date:** 2026-03-25
**Status:** Approved
**Skill location:** `~/.claude/skills/stanobj/`

---

## Purpose

Convert heterogeneous single-cell transcriptomics datasets from many source formats into a canonical, analysis-ready `.h5ad` representation. This is a data standardization and semantic harmonization task, not just file-format conversion.

Prioritizes correctness, traceability, and preservation of biological meaning over aggressive automation.

## Scope

### What this skill does

- Detect source format (with compression/archive support)
- Read datasets from CSV, TSV, MTX, 10x h5, generic HDF5, Seurat RDS, SingleCellExperiment, existing h5ad, loom
- Infer and validate matrix orientation
- Classify matrix semantic type (counts, normalized, log1p, scaled, unknown)
- Extract and standardize expression matrix, cell metadata, gene metadata, embeddings
- Produce canonical AnnData with structured provenance
- Generate machine-readable report and human-readable audit log
- Surface ambiguities explicitly — never silently guess

### What this skill does NOT do

- Gene ID conversion or mapping (that is `stangene`)
- Inspect/compare/merge existing h5ad files (that is `scrna-reader`)
- Biological analysis (PCA, clustering, trajectory, QC)
- Batch processing of multiple datasets in one invocation

### Relationship to other skills

- **`scrna-reader`**: Inspects, compares, merges existing h5ad. `stanobj` converts *into* canonical h5ad. If an h5ad needs re-standardization, `stanobj` handles it. If user wants to read/describe an h5ad, that's `scrna-reader`.
- **`stangene`**: Handles gene ID conversion/mapping. `stanobj` detects and preserves gene IDs, ensures uniqueness, but does not convert between ID types.

---

## Architecture

### Approach: Orchestrator + Modular Backends (Hybrid execution)

A main `stanobj.py` orchestrator delegates to format-specific reader modules and shared validation/standardization modules. It runs autonomously through the happy path but exits with structured decision requests when it hits ambiguity. Claude intervenes at decision points, then re-invokes.

### File Structure

```
~/.claude/skills/stanobj/
├── SKILL.md
├── scripts/
│   ├── stanobj.py              # Main orchestrator CLI
│   ├── readers/
│   │   ├── __init__.py
│   │   ├── csv_reader.py       # CSV/TSV (most ambiguous)
│   │   ├── mtx_reader.py       # Matrix Market / 10x triplet
│   │   ├── h5_reader.py        # 10x h5 + generic HDF5
│   │   ├── h5ad_reader.py      # Existing h5ad re-standardization
│   │   ├── loom_reader.py      # Loom format
│   │   ├── seurat_reader.R     # R subprocess for Seurat RDS
│   │   └── sce_reader.R        # R subprocess for SingleCellExperiment
│   ├── detection.py            # Format detection + matrix type inference
│   ├── validation.py           # All validation checks
│   ├── standardize.py          # Schema normalization, metadata harmonization
│   ├── report.py               # JSON report + audit log generation
│   └── utils.py                # Shared utilities
├── references/
│   ├── canonical-schema.md     # Target AnnData schema reference
│   ├── format-guide.md         # Format-specific notes and gotchas
│   └── decision-points.md      # All decision points documented
└── examples/
    └── sample-report.json      # Example conversion report
```

---

## Skill Metadata

```yaml
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
```

---

## CLI Interface

```bash
# Basic invocation
python stanobj.py <input_path> -o <output.h5ad>

# With decisions pre-supplied (from Claude re-invocation)
python stanobj.py <input_path> -o <output.h5ad> --decision assay=RNA --decision matrix_type=counts

# With explicit format override (skip auto-detection)
python stanobj.py <input_path> -o <output.h5ad> --format seurat_rds
```

- `input_path`: a file or a directory (for 10x triplet: directory containing matrix.mtx + barcodes.tsv + features.tsv)
- `-o`: required output path for the `.h5ad`
- Report and audit log written to `<output_stem>_report.json` and `<output_stem>_audit.log` alongside the output h5ad

---

## Orchestrator Pipeline

```
1. Decompress/extract (if compressed/archived)
2. detect_format(input_path)
3. read_source(input_path, format) → raw AnnData + source_metadata
4. infer_and_validate(adata) → orientation, matrix_type, feature_types
5. standardize(adata, source_metadata) → canonical AnnData
6. validate_final(adata) → pass/fail + warnings
7. write_output(adata, output_path) → .h5ad + report.json + audit.log
```

Each step can either succeed (continue) or yield a decision request.

---

## Decision Protocol

### Exit codes

- `0` — success, outputs written
- `1` — fatal error (unrecoverable)
- `10` — decision needed, structured JSON on stdout

### Decision request format (exit 10)

```json
{
  "status": "decision_needed",
  "decision_type": "assay_selection",
  "context": "Seurat object contains 3 assays: RNA (32738 features), SCT (3000 features), ADT (142 features)",
  "options": ["RNA", "SCT", "ADT"],
  "recommendation": "RNA",
  "reason": "RNA has the most features and contains raw counts",
  "partial_state": "/tmp/stanobj_abc123/state.pkl"
}
```

Claude re-invokes with `--decision assay_selection=RNA`. The orchestrator re-runs from scratch by default (most conversions are fast). For expensive operations (R subprocess reads), the orchestrator may optionally save intermediate state to `partial_state` and accept a `--resume <path>` flag to skip the expensive read step. Resume is best-effort — if the state file is missing or incompatible, the orchestrator falls back to a full re-run.

### Known decision points

| Decision Type | When | Options |
|---|---|---|
| `format_detection` | Cannot determine source format | format names |
| `assay_selection` | Seurat/SCE with multiple assays | assay names |
| `matrix_orientation` | CSV/TSV ambiguous rows vs cols | `cells_x_genes`, `genes_x_cells` |
| `matrix_type` | Can't confidently classify | `counts`, `normalized`, `log1p`, `scaled`, `unknown` |
| `layer_selection` | Multiple candidate layers, unclear which is primary | layer names |
| `modality_filter` | Mixed feature types detected | `rna_only`, `keep_all` |
| `id_column` | CSV/TSV first column ambiguous | `row_ids`, `data_column` |

---

## Compression & Archive Handling

### Pre-processing step before format detection

**Transparent decompression** (handled inline by readers):
- `.csv.gz`, `.tsv.gz` — pandas handles natively
- `matrix.mtx.gz` — scipy `mmread` handles natively
- `barcodes.tsv.gz`, `features.tsv.gz`, `genes.tsv.gz` — pandas reads `.gz` natively

**Archive extraction** (requires temp directory):
- `.tar.gz` / `.tgz` — extract to temp dir → detect format on contents
- `.tar.bz2` / `.tbz2` — extract to temp dir → detect format on contents
- `.tar` — extract to temp dir → detect format on contents
- `.zip` — extract to temp dir → detect format on contents

**Single-file decompression** (requires temp file for non-seekable formats):
- `.h5.gz` / `.h5ad.gz` — decompress to temp file (HDF5 requires seekable file)
- `.rds.gz` — decompress to temp file before R reads it
- `.gz` (standalone) — decompress, detect inner format
- `.bz2` (standalone) — decompress, detect inner format

**Detection flow:**
```
input_path
  → is it an archive (.tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .zip)?
      yes → extract to temp dir → detect format on contents
      no  → strip .gz/.bz2 from extension for format detection
            → pass compressed path to reader (most handle .gz natively)
            → if reader needs seekable file (h5/h5ad/rds), decompress to temp first
```

**Cleanup:** temp directories removed after successful conversion. On failure, preserved and path noted in error output.

---

## Format Detection & Reader Modules

### Auto-detection logic

```
.h5ad / .h5ad.gz         → h5ad_reader
.rds / .rds.gz           → seurat_reader (try R subprocess)
.h5 / .hdf5 / .h5.gz    → inspect HDF5 groups:
                             has "matrix/" group → 10x h5
                             has "X/" or "obs/" → h5ad-like
                             other → generic h5 (inspect + decision)
.mtx / .mtx.gz /         → mtx_reader (look for barcodes.tsv + features.tsv)
  directory with matrix.mtx
.csv / .csv.gz            → csv_reader
.tsv / .tsv.gz            → csv_reader (tab delimiter)
.loom                     → loom_reader
```

`--format` flag bypasses auto-detection entirely.

### Reader modules

Each reader returns a uniform intermediate: a raw AnnData object + a `source_metadata` dict.

**`csv_reader.py`** — Most ambiguous format:
1. Read first few rows: detect delimiter, header presence, first-column-as-index
2. Heuristic checks: row labels gene-like? barcode-like? values integer-like?
3. If orientation ambiguous → decision request
4. Load into sparse matrix if large (threshold ~10k rows)

**`mtx_reader.py`** — Structured, some gotchas:
1. Read `matrix.mtx` via scipy
2. Parse `barcodes.tsv` and `features.tsv` / `genes.tsv`
3. Handle 2-column vs 3-column features file
4. Tag `feature_type` from column 3 if present
5. Matrix Market is genes x cells by convention → transpose to cells x genes

**`h5_reader.py`** — Two sub-paths:
1. 10x HDF5: look for `matrix/` group with `barcodes`, `data`, `indices`, `indptr`, `features`
2. Generic HDF5: list top-level groups, attempt to identify matrix datasets and metadata. If unrecognized → decision request with HDF5 tree dump

**`h5ad_reader.py`** — Re-standardization path:
1. Read with `anndata.read_h5ad`
2. Don't assume already canonical
3. Run same validation/standardization pipeline as every other format

**`seurat_reader.R`** — R subprocess:
1. Load `.rds`, verify Seurat object
2. List assays, default assay, available slots (counts/data/scale.data)
3. List reductions, metadata columns
4. Output inventory as JSON to stdout
5. If multiple assays → exit with decision request JSON
6. Once assay chosen: export counts matrix as MTX + barcodes.tsv + features.tsv + metadata.csv + reductions.csv to temp directory
7. Python picks up via mtx_reader + metadata merge

**`sce_reader.R`** — Same pattern:
1. Load `.rds`, verify SingleCellExperiment
2. List assays, colData, rowData, reducedDims, altExps
3. Export chosen assay as MTX + metadata to temp directory

**`loom_reader.py`** — Via loompy:
1. Read with `loompy.connect`
2. Matrix is genes x cells in loom → transpose
3. Extract row/col attributes as var/obs metadata

**R fallback:** if `Rscript` not available or R script fails, output human-readable instructions explaining how to convert manually (e.g., `SeuratDisk::SaveH5Seurat(obj, 'output.h5seurat')` in R, then re-invoke stanobj on the h5seurat file).

---

## Matrix Orientation Validation

**Known-orientation formats** (no guessing):
- MTX / 10x: genes x cells → always transpose
- Loom: genes x cells → always transpose
- Seurat/SCE export: R scripts export in known orientation
- h5ad: cells x genes (verify)

**Ambiguous formats** (CSV/TSV, generic HDF5) — heuristic checks on shape `(n_rows, n_cols)`:
1. Row labels match gene patterns + col labels match barcode patterns → genes x cells → transpose
2. Col labels match gene patterns + row labels match barcode patterns → cells x genes → keep
3. `n_rows >> n_cols` and `n_cols` in typical gene range (15k-60k) → rows are cells
4. `n_cols >> n_rows` and `n_rows` in typical gene range → cols are cells, transpose
5. Still ambiguous → decision request `matrix_orientation`

Any transposition recorded in report.

---

## Matrix Type Classification

Classify as: `counts`, `normalized`, `log1p`, `scaled`, `unknown`.

Sampling: up to 1000 cells, seed 42.

### Decision tree

```
Has negative values?
  yes → "scaled"
  no  →
    All values integer-like (within float tolerance 1e-6)?
      yes → max > 10?
        yes → "counts"
        no  → "unknown"
      no  →
        max < 20 and most values < 15?
          yes → "log1p"
          no  →
            Row sums CV < 0.01 and max > 100?
              yes → "normalized" (CPM/TPM)
              no  → "unknown"
```

### Override sources (trusted over heuristics)

- Seurat slot: `counts` → counts, `data` → log1p, `scale.data` → scaled
- SCE assay name: `counts` → counts, `logcounts` → log1p, `normcounts` → normalized
- h5ad layer name: `counts` → counts (still verify with heuristics as sanity check)
- If source metadata contradicts heuristics → warn but trust source metadata
- Low confidence → decision request `matrix_type`

---

## Canonical Output Schema

### X and layers

- Raw counts exist + normalized/log1p also exists → `X = normalized/log1p`, `layers["counts"] = raw`
- Only raw counts → `X = counts`, `layers["counts"] = counts`
- Only processed data → `X = processed`, no `layers["counts"]`, warn counts missing
- Decision recorded in `adata.uns["stanobj"]["x_contents"]`

### obs standardization

- `obs_names`: unique cell identifiers
- `cell_id`: original barcode/identifier
- `dataset`: derived from input filename
- Column name mapping (original columns preserved, new standardized columns added):
  - `celltype` / `CellType` / `cell_type_label` / `annotation` → `cell_type`
  - `orig.ident` / `sample_id` / `sampleName` → `sample`
  - `patient` / `donor_id` / `individual` → `donor`
  - `disease` / `treatment` / `experimental_condition` → `condition`
- Non-unique `obs_names` → prepend dataset name, store originals in `cell_id`

### var standardization

- `var_names`: unique gene identifiers (from source)
- `gene_symbol`: preserved if available
- `gene_id`: Ensembl ID if available
- `feature_type`: from source (e.g. "Gene Expression", "Antibody Capture")
- `original_gene_name`: pre-uniquification name if duplicates resolved
- Duplicate `var_names`: make unique with suffix (`-1`, `-2`), store original in `original_gene_name`

### obsm (embeddings)

- Preserve PCA, UMAP, tSNE with standard keys: `X_pca`, `X_umap`, `X_tsne`
- Map common source names: `pca` → `X_pca`, `umap` → `X_umap`

### uns (provenance)

```python
adata.uns["stanobj"] = {
    "source_path": str,
    "source_format": str,
    "conversion_timestamp": str,      # ISO 8601
    "matrix_type": str,
    "x_contents": str,
    "transposed": bool,
    "assay_selected": str | None,
    "layer_selected": str | None,
    "modality_filter": str | None,
    "decompressed": bool,
    "obs_name_strategy": str,
    "var_name_strategy": str,
    "decisions_made": dict,
    "warnings": list[str],
    "version": "1.0.0"
}
```

### Sparsity

- Preserve sparse matrices as sparse (csr_matrix or csc_matrix)
- Dense + large (>10k cells) → convert to sparse, log it
- Dense + small → keep as-is

---

## Validation Checks

### Fatal (block output)

- Matrix is 2D
- `n_obs > 0` and `n_vars > 0`
- Matrix shape matches `len(obs_names)` x `len(var_names)`
- `obs_names` are unique
- `var_names` are unique
- Matrix type has been classified (even if `unknown`)

### Warning (logged, output proceeds)

- Counts layer declared but values not nonneg integer-like
- Zero metadata columns in obs or var
- Extremely high sparsity (>99.9%)
- Very few cells (<50) or genes (<100)
- Embeddings dimension mismatch → drop + warn
- Non-RNA features present without `feature_type` column
- Format resolved by heuristics only

---

## Output Files

```
<output_path>.h5ad              # Canonical AnnData
<output_stem>_report.json       # Machine-readable conversion report
<output_stem>_audit.log         # Human-readable audit log
```

### report.json fields

```json
{
  "stanobj_version": "1.0.0",
  "source_path": "/data/patient1.rds",
  "source_format": "seurat_rds",
  "decompressed": false,
  "reader_used": "seurat_reader",
  "matrix_orientation_before": "cells_x_genes",
  "transposed": false,
  "matrix_type": "counts",
  "matrix_type_confidence": "high",
  "matrix_type_source": "seurat_counts_slot",
  "x_contents": "counts",
  "layers": ["counts"],
  "main_assay": "RNA",
  "raw_counts_found": true,
  "feature_types_present": ["Gene Expression"],
  "rna_only_subset_applied": false,
  "gene_identifier_type": "symbol_human",
  "duplicate_genes_resolved": 3,
  "obs_name_strategy": "original_unique",
  "var_name_strategy": "symbol_with_dedup",
  "n_cells_before": 8500,
  "n_genes_before": 32738,
  "n_cells_after": 8500,
  "n_genes_after": 32738,
  "embeddings_preserved": ["X_pca", "X_umap"],
  "embeddings_dropped": [],
  "obs_columns_standardized": {"orig.ident": "sample", "celltype": "cell_type"},
  "decisions_made": {"assay_selection": "RNA"},
  "warnings": ["3 duplicate gene symbols resolved with suffix"],
  "errors": [],
  "conversion_timestamp": "2026-03-25T16:45:00Z"
}
```

### audit.log format

```
=== stanobj conversion audit ===
Source: <path> (<format>)
Output: <path>
Timestamp: <ISO 8601>

--- Detection ---
<format details, assays found, slots available>

--- Standardization ---
<X contents, orientation, matrix type, gene/cell ID handling, embeddings, metadata mapping>

--- Warnings ---
<each warning on its own line>

--- Validation ---
<pass/fail summary>

Shape: <n_cells> cells x <n_genes> genes
```

---

## SKILL.md Workflow (Claude orchestration)

1. Identify the input file/directory from user's request
2. Ask user for output path (required)
3. Run `python stanobj.py <input> -o <output>`
4. If exit 0 → read audit log, present summary to user
5. If exit 10 → read decision JSON, present options to user (or decide if obvious), re-invoke with `--decision`
6. If exit 1 → read error, report to user
7. After success, suggest running `scrna-reader` to verify output if desired

---

## Environment Requirements

**Python** (available): anndata, scanpy, h5py, scipy, loompy, mudata, pandas, numpy

**R** (available, for Seurat/SCE): Seurat, SeuratObject, SeuratDisk, SingleCellExperiment, scater

**R bridge**: No rpy2/zellkonverter. Strategy: R subprocess scripts export to intermediate files (MTX + metadata), Python picks them up. Fallback: manual conversion instructions.

---

## Warning Policy

Warn loudly when:
- Matrix type cannot be determined confidently
- Matrix orientation is ambiguous
- Multiple assays exist and correct one is uncertain
- Counts are missing
- Metadata and expression matrix may be misaligned
- Features include multiple modalities
- Gene identifier type is mixed or unclear
- Duplicate identifiers fixed automatically
- Input HDF5 structure is custom and partially inferred
- Compression/archive was handled (record original filename)

Never hide uncertainty.

---

## Decision Priority

When multiple actions are possible:
1. Preserve source information
2. Avoid semantic corruption
3. Keep output interoperable
4. Be explicit about uncertainty
5. Automate only when confidence is high
