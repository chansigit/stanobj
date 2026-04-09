"""Reader for loom files using h5py directly (more robust than loompy)."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .base import ReaderResult
except ImportError:
    from base import ReaderResult

# Keys recognised as gene-name identifiers in row_attrs
_GENE_KEYS = {"Gene", "gene"}

# Keys recognised as cell-ID identifiers in col_attrs
_CELL_KEYS = {"CellID", "cellid", "cell_id"}


def _decode_array(arr: np.ndarray) -> np.ndarray:
    """Decode bytes elements to str; pass through non-bytes arrays."""
    if arr.dtype.kind in ("S", "O"):
        return np.array(
            [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
        )
    return arr


def _read_attrs_group(group: h5py.Group) -> dict[str, np.ndarray]:
    """Read all datasets from an HDF5 group, decoding bytes to str."""
    attrs: dict[str, np.ndarray] = {}
    for key in group:
        arr = group[key][:]
        attrs[key] = _decode_array(arr)
    return attrs


def read_loom(path: str, decisions: Optional[dict] = None) -> ReaderResult:
    """Read a loom file using h5py directly.

    Parameters
    ----------
    path : str
        Path to the ``.loom`` file.
    decisions : dict, optional
        Reserved for future decision-based interactivity.

    Returns
    -------
    ReaderResult
        With ``adata`` in cells-by-genes orientation and ``source_meta``
        describing the conversion.
    """
    with h5py.File(path, "r") as f:
        # 1. Read matrix (genes x cells)
        matrix = f["matrix"][:]

        # 2-3. Read row_attrs and col_attrs
        row_attrs = _read_attrs_group(f["row_attrs"])
        col_attrs = _read_attrs_group(f["col_attrs"])

    n_genes, n_cells = matrix.shape

    # 4. Transpose to cells x genes and make sparse CSR
    X = sparse.csr_matrix(matrix.T.astype(np.float32))

    # 5. Build var from row_attrs
    gene_key = None
    for k in _GENE_KEYS:
        if k in row_attrs:
            gene_key = k
            break

    if gene_key is not None:
        var_names = row_attrs.pop(gene_key)
    else:
        var_names = np.arange(n_genes).astype(str)

    var = pd.DataFrame(row_attrs, index=var_names)

    # 6. Build obs from col_attrs
    cell_key = None
    for k in _CELL_KEYS:
        if k in col_attrs:
            cell_key = k
            break

    if cell_key is not None:
        obs_names = col_attrs.pop(cell_key)
    else:
        obs_names = np.arange(n_cells).astype(str)

    obs = pd.DataFrame(col_attrs, index=obs_names)

    # 7. Assemble AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)

    source_meta = {
        "source_format": "loom",
        "reader_used": "loom_reader",
        "matrix_orientation_before": "genes_x_cells",
        "transposed": True,
        "raw_counts_found": False,
        "feature_types_present": ["Gene Expression"],
        "matrix_type_hint": None,
        "decompressed": False,
        "warnings": [],
    }

    return ReaderResult(adata=adata, source_meta=source_meta)
