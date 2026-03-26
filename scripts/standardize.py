"""Standardization helpers for obs, var, obsm, layers, and provenance."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.utils import iso_now, make_names_unique

# ---------------------------------------------------------------------------
# Column-name mapping: source variants -> canonical name
# ---------------------------------------------------------------------------

OBS_COLUMN_MAP: Dict[str, str] = {
    # cell_type variants
    "celltype": "cell_type",
    "CellType": "cell_type",
    "Celltype": "cell_type",
    "cell_type_label": "cell_type",
    "annotation": "cell_type",
    "cluster_label": "cell_type",
    # sample variants
    "orig.ident": "sample",
    "sample_id": "sample",
    "sampleName": "sample",
    "Sample": "sample",
    "SampleID": "sample",
    # donor variants
    "patient": "donor",
    "donor_id": "donor",
    "individual": "donor",
    "Patient": "donor",
    "subject": "donor",
    "subject_id": "donor",
    # condition variants
    "disease": "condition",
    "treatment": "condition",
    "experimental_condition": "condition",
    "Disease": "condition",
    "diagnosis": "condition",
    # batch variants
    "batch_id": "batch",
    "Batch": "batch",
}

# ---------------------------------------------------------------------------
# obsm key mapping
# ---------------------------------------------------------------------------

_OBSM_KEY_MAP: Dict[str, str] = {
    "pca": "X_pca",
    "PCA": "X_pca",
    "umap": "X_umap",
    "UMAP": "X_umap",
    "tsne": "X_tsne",
    "tSNE": "X_tsne",
}

# Layer names that indicate processed / log-transformed data
_PROCESSED_LAYER_NAMES = {"log1p", "logcounts", "data", "normalized", "normcounts"}

# Layer names that indicate raw counts
_COUNTS_LAYER_NAMES = {"counts", "raw_counts"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def standardize_obs(
    obs: pd.DataFrame,
    dataset_name: str,
    make_unique: bool = False,
) -> pd.DataFrame:
    """Standardize an obs DataFrame.

    - Copies *obs* (never mutates the original).
    - Adds ``cell_id`` column from the original index.
    - Adds ``dataset`` column.
    - Maps column-name variants to canonical names (without overwriting
      existing canonical columns).
    - Optionally makes the index unique.
    """
    result = obs.copy()

    # Record original index as cell_id
    result["cell_id"] = result.index.values
    result["dataset"] = dataset_name

    # Map variant column names -> canonical names
    for source_col, canon_col in OBS_COLUMN_MAP.items():
        if source_col in result.columns and canon_col not in result.columns:
            result[canon_col] = result[source_col]

    # Uniquify index if requested or if there are duplicates
    if make_unique or not result.index.is_unique:
        names = list(result.index.astype(str))
        unique_names, _ = make_names_unique(names)
        result.index = pd.Index(unique_names)

        # If still not unique, prepend dataset_name and try again
        if not result.index.is_unique:
            names = [f"{dataset_name}_{n}" for n in result.index]
            unique_names, _ = make_names_unique(names)
            result.index = pd.Index(unique_names)

    return result


def standardize_var(
    var: pd.DataFrame,
    return_rename_map: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """Standardize a var DataFrame.

    - Copies *var* (never mutates the original).
    - If ``gene_symbol`` is not among the columns, adds it from the index.
    - If duplicates exist in the index, makes them unique and stores
      the originals in an ``original_gene_name`` column.
    """
    result = var.copy()

    # Add gene_symbol from index if missing
    if "gene_symbol" not in result.columns:
        result["gene_symbol"] = result.index.values

    # Handle duplicate gene names in the index
    rename_map: dict = {}
    if not result.index.is_unique:
        original_names = list(result.index.astype(str))
        result["original_gene_name"] = original_names
        unique_names, rename_map = make_names_unique(original_names)
        result.index = pd.Index(unique_names)

    if return_rename_map:
        return result, rename_map
    return result


def standardize_obsm(obsm: dict) -> dict:
    """Standardize obsm keys to AnnData conventions.

    Maps common embedding names (pca, UMAP, tSNE, etc.) to their
    ``X_pca``, ``X_umap``, ``X_tsne`` canonical forms.  Keys that are
    already standard or unknown are passed through unchanged.
    """
    result: dict = {}
    for key, value in obsm.items():
        new_key = _OBSM_KEY_MAP.get(key, key)
        result[new_key] = value
    return result


def assign_layers(
    adata: "anndata.AnnData",
    matrix_type: str,
    source_layers: dict,
) -> "anndata.AnnData":
    """Organise X and layers based on *matrix_type*.

    Parameters
    ----------
    adata
        AnnData whose ``.X`` is the primary matrix.
    matrix_type
        Classification of the primary matrix (e.g. ``"counts"``).
    source_layers
        Additional layers from the source file keyed by name.

    Returns
    -------
    adata
        The same object, mutated in place for efficiency.
    """
    if matrix_type == "counts":
        # Store raw counts in layers["counts"]
        adata.layers["counts"] = adata.X.copy()

        # Look for a processed layer to put in X
        for name in _PROCESSED_LAYER_NAMES:
            if name in source_layers:
                adata.X = source_layers.pop(name).copy()
                adata.uns["stanobj_x_contents"] = name
                break
        else:
            # No processed layer found — X stays as counts
            adata.uns["stanobj_x_contents"] = "counts"
    else:
        # X is already processed
        adata.uns["stanobj_x_contents"] = matrix_type

        # Look for raw counts in source_layers
        for name in _COUNTS_LAYER_NAMES:
            if name in source_layers:
                adata.layers["counts"] = source_layers.pop(name).copy()
                break

    # Copy remaining source layers
    for name, data in source_layers.items():
        adata.layers[name] = data

    return adata


def add_provenance(
    adata: "anndata.AnnData",
    source_meta: dict,
    matrix_classification: str,
    decisions: List[str],
    all_warnings: List[str],
) -> "anndata.AnnData":
    """Attach provenance metadata in ``adata.uns["stanobj"]``.

    Parameters
    ----------
    adata
        The AnnData object to annotate.
    source_meta
        Dict with keys like ``source_path``, ``source_format``, ``transposed``,
        ``assay_selected``, ``layer_selected``, ``modality_filter``,
        ``decompressed``, ``obs_name_strategy``, ``var_name_strategy``.
    matrix_classification
        The classification string for the matrix (e.g. ``"counts"``).
    decisions
        List of human-readable decision strings.
    all_warnings
        List of warning strings.
    """
    x_contents = adata.uns.pop("stanobj_x_contents", matrix_classification)

    adata.uns["stanobj"] = {
        "source_path": source_meta.get("source_path", ""),
        "source_format": source_meta.get("source_format", ""),
        "conversion_timestamp": iso_now(),
        "matrix_type": matrix_classification,
        "x_contents": x_contents,
        "transposed": source_meta.get("transposed", False),
        "assay_selected": source_meta.get("assay_selected", None),
        "layer_selected": source_meta.get("layer_selected", None),
        "modality_filter": source_meta.get("modality_filter", None),
        "decompressed": source_meta.get("decompressed", False),
        "obs_name_strategy": source_meta.get("obs_name_strategy", None),
        "var_name_strategy": source_meta.get("var_name_strategy", None),
        "decisions_made": decisions,
        "warnings": all_warnings,
        "version": "1.0.0",
    }

    return adata


def ensure_sparse(
    adata: "anndata.AnnData",
    threshold: int = 10_000,
) -> bool:
    """Convert ``adata.X`` to CSR if it is dense and large enough.

    Parameters
    ----------
    adata
        The AnnData object (mutated in place).
    threshold
        Minimum number of observations to trigger conversion.

    Returns
    -------
    bool
        True if conversion was performed.
    """
    if not sparse.issparse(adata.X) and adata.n_obs >= threshold:
        adata.X = sparse.csr_matrix(adata.X)
        return True
    return False
