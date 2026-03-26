"""Reader for HDF5 files: 10x Genomics .h5 and generic/unrecognised HDF5."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .base import DecisionNeeded, ReaderResult
except ImportError:
    from base import DecisionNeeded, ReaderResult


# ---------------------------------------------------------------------------
# 10x Genomics HDF5
# ---------------------------------------------------------------------------


def read_10x_h5(path: str, decisions: Optional[dict] = None) -> ReaderResult:
    """Read a 10x Genomics HDF5 file (contains ``matrix/`` group).

    The file stores a CSC sparse matrix in genes-by-cells orientation.
    This reader transposes to cells-by-genes before returning.

    Parameters
    ----------
    path : str
        Path to a 10x HDF5 file (e.g. ``filtered_feature_bc_matrix.h5``).
    decisions : dict, optional
        Reserved for future decision-based interactivity.

    Returns
    -------
    ReaderResult
    """
    with h5py.File(path, "r") as f:
        grp = f["matrix"]

        # 1. Reconstruct CSC sparse matrix (genes x cells) -----------------
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(grp["shape"][:])
        mtx = sparse.csc_matrix((data, indices, indptr), shape=shape)

        # 2. Read barcodes -------------------------------------------------
        barcodes_raw = grp["barcodes"][:]
        barcodes = [b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    for b in barcodes_raw]

        # 3. Read features -------------------------------------------------
        feat = grp["features"]

        gene_ids_raw = feat["id"][:]
        gene_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in gene_ids_raw]

        gene_names_raw = feat["name"][:]
        gene_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                      for x in gene_names_raw]

        if "feature_type" in feat:
            ft_raw = feat["feature_type"][:]
            feature_types = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                             for x in ft_raw]
        else:
            feature_types = ["Gene Expression"] * len(gene_ids)

    # 4. Transpose to cells x genes ----------------------------------------
    X = mtx.T.tocsr()

    # 5. Build var and obs -------------------------------------------------
    var = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": gene_names,
            "feature_type": feature_types,
        },
        index=gene_names,
    )

    obs = pd.DataFrame(index=barcodes)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # 6. Source metadata ----------------------------------------------------
    feature_types_present = sorted(set(feature_types))

    source_meta = {
        "source_format": "10x_h5",
        "reader_used": "h5_reader:10x",
        "matrix_orientation_before": "genes_x_cells",
        "transposed": True,
        "raw_counts_found": True,
        "feature_types_present": feature_types_present,
        "matrix_type_hint": "counts",
        "decompressed": False,
        "warnings": [],
    }

    return ReaderResult(adata=adata, source_meta=source_meta)


# ---------------------------------------------------------------------------
# Generic / unrecognised HDF5
# ---------------------------------------------------------------------------

_MAX_TREE_ITEMS = 50


def read_generic_h5(path: str, decisions: Optional[dict] = None) -> ReaderResult:
    """Attempt to read an unrecognised HDF5 file.

    Walks the HDF5 tree and raises :class:`DecisionNeeded` so that the
    caller (or an interactive user) can decide how to proceed.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    decisions : dict, optional
        Reserved for future decision-based interactivity.

    Returns
    -------
    ReaderResult
        (never reached — always raises)

    Raises
    ------
    DecisionNeeded
        With ``decision_type="format_detection"`` and a summary of the
        HDF5 tree structure.
    """
    tree_lines: list[str] = []

    def _visitor(name: str, obj: h5py.HLObject) -> None:
        if len(tree_lines) >= _MAX_TREE_ITEMS:
            return
        if isinstance(obj, h5py.Dataset):
            tree_lines.append(
                f"  dataset: {name}  shape={obj.shape}  dtype={obj.dtype}"
            )
        elif isinstance(obj, h5py.Group):
            tree_lines.append(f"  group: {name}/")

    with h5py.File(path, "r") as f:
        f.visititems(_visitor)

    tree_summary = "\n".join(tree_lines) if tree_lines else "(empty file)"
    context = (
        f"Unrecognised HDF5 structure in {path}.\n"
        f"HDF5 tree (up to {_MAX_TREE_ITEMS} items):\n{tree_summary}"
    )

    raise DecisionNeeded(
        decision_type="format_detection",
        context=context,
        options=["10x_h5", "h5ad", "skip"],
        reason="Could not automatically determine the HDF5 sub-format.",
    )
