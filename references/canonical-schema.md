# Canonical h5ad Schema Reference

## Matrix Conventions

### `adata.X`
Main analysis matrix. Convention:
- If raw counts + processed both available: `X = processed`, `layers["counts"] = raw`
- If only raw counts: `X = counts`, `layers["counts"] = counts`
- If only processed: `X = processed`, warn that counts are missing

Always documented in `adata.uns["stanobj"]["x_contents"]`.

### `adata.layers["counts"]`
Raw count matrix when available. Values should be non-negative and approximately integer.

## Observation Axis (obs)

`adata.obs_names` must be unique cell identifiers.

### Standard Columns
| Column | Description | Source Variants |
|--------|-------------|-----------------|
| `cell_id` | Original cell identifier / barcode | Always added |
| `dataset` | Dataset name (from filename) | Always added |
| `cell_type` | Author-provided cell annotation | celltype, CellType, cell_type_label, annotation |
| `sample` | Sample identifier | orig.ident, sample_id, sampleName |
| `donor` | Donor / patient / individual | patient, donor_id, individual, subject |
| `condition` | Disease / treatment / condition | disease, treatment, diagnosis |
| `batch` | Batch label | batch_id, Batch |

Original columns are never deleted. Standardized columns are added alongside.

## Variable Axis (var)

`adata.var_names` must be unique gene identifiers.

### Standard Columns
| Column | Description |
|--------|-------------|
| `gene_symbol` | Gene symbol (always present) |
| `gene_id` | Ensembl ID if available |
| `feature_type` | e.g. Gene Expression, Antibody Capture |
| `original_gene_name` | Pre-uniquification name (only if duplicates existed) |

## Embeddings (obsm)

Standard keys: `X_pca`, `X_umap`, `X_tsne`

Source names are mapped: `pca` -> `X_pca`, `umap` -> `X_umap`

## Provenance (uns)

```python
adata.uns["stanobj"] = {
    "source_path": str,
    "source_format": str,
    "conversion_timestamp": str,
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
