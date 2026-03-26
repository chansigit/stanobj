"""Reader for Matrix Market (.mtx) triplet directories (10x Genomics style)."""

from __future__ import annotations

import os
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

try:
    from .base import ReaderResult
except ImportError:
    from base import ReaderResult


def _find_file(directory: str, candidates: list[str]) -> Optional[str]:
    """Return the first existing file from *candidates* inside *directory*."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def read_mtx(path: str, decisions: Optional[dict] = None) -> ReaderResult:
    """Read a Matrix Market triplet directory or single .mtx file.

    Parameters
    ----------
    path : str
        Path to a directory containing matrix.mtx, barcodes.tsv, and
        features.tsv (or genes.tsv), OR a direct path to a .mtx file.
    decisions : dict, optional
        Reserved for future decision-based interactivity.

    Returns
    -------
    ReaderResult
        Contains the AnnData object and source metadata.
    """
    # 1. Resolve directory --------------------------------------------------
    if os.path.isfile(path):
        directory = os.path.dirname(os.path.abspath(path))
    else:
        directory = os.path.abspath(path)

    # 2. Locate required files ----------------------------------------------
    matrix_path = _find_file(directory, ["matrix.mtx.gz", "matrix.mtx"])
    barcodes_path = _find_file(directory, ["barcodes.tsv.gz", "barcodes.tsv"])
    features_path = _find_file(
        directory,
        ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"],
    )

    # 3. Validate all files exist -------------------------------------------
    missing = []
    if matrix_path is None:
        missing.append("matrix.mtx[.gz]")
    if barcodes_path is None:
        missing.append("barcodes.tsv[.gz]")
    if features_path is None:
        missing.append("features.tsv[.gz] or genes.tsv[.gz]")
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {directory}: {', '.join(missing)}"
        )

    # Track whether any gzipped file was used
    decompressed = any(
        p.endswith(".gz") for p in (matrix_path, barcodes_path, features_path)
    )

    # 4. Read matrix --------------------------------------------------------
    mtx = mmread(matrix_path)
    mtx = sparse.csr_matrix(mtx)

    # 5. Read barcodes ------------------------------------------------------
    barcodes_df = pd.read_csv(
        barcodes_path, sep="\t", header=None, names=["barcode"]
    )

    # 6. Read features ------------------------------------------------------
    features_df = pd.read_csv(features_path, sep="\t", header=None)
    n_cols = features_df.shape[1]

    if n_cols >= 3:
        features_df.columns = ["gene_id", "gene_symbol", "feature_type"] + [
            f"col_{i}" for i in range(3, n_cols)
        ]
    elif n_cols == 2:
        features_df.columns = ["gene_id", "gene_symbol"]
        features_df["feature_type"] = "Gene Expression"
    else:
        features_df.columns = ["gene_id"]
        features_df["gene_symbol"] = features_df["gene_id"]
        features_df["feature_type"] = "Gene Expression"

    # 7. Validate shape (genes x cells) -------------------------------------
    n_genes, n_cells = mtx.shape
    if n_genes != len(features_df):
        raise ValueError(
            f"Matrix has {n_genes} rows but features file has "
            f"{len(features_df)} entries."
        )
    if n_cells != len(barcodes_df):
        raise ValueError(
            f"Matrix has {n_cells} columns but barcodes file has "
            f"{len(barcodes_df)} entries."
        )

    # 8. Transpose to cells x genes ----------------------------------------
    X = mtx.T.tocsr()

    # 9. Build var DataFrame ------------------------------------------------
    var = pd.DataFrame(
        {
            "gene_id": features_df["gene_id"].values,
            "gene_symbol": features_df["gene_symbol"].values,
            "feature_type": features_df["feature_type"].values,
        },
        index=features_df["gene_symbol"].values,
    )

    # 10. Build obs DataFrame -----------------------------------------------
    obs = pd.DataFrame(index=barcodes_df["barcode"].values)

    # 11. Create AnnData ----------------------------------------------------
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # 12. Build source metadata ---------------------------------------------
    feature_types_present = sorted(features_df["feature_type"].unique().tolist())

    warnings_list: list[str] = []

    source_meta = {
        "source_format": "mtx",
        "reader_used": "mtx_reader",
        "matrix_orientation_before": "genes_x_cells",
        "transposed": True,
        "raw_counts_found": True,
        "feature_types_present": feature_types_present,
        "matrix_type_hint": "counts",
        "decompressed": decompressed,
        "warnings": warnings_list,
    }

    return ReaderResult(adata=adata, source_meta=source_meta)
