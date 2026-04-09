"""Reader for CSV/TSV expression matrices."""

from __future__ import annotations

import os
import re
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .base import DecisionNeeded, ReaderResult
except ImportError:
    from base import DecisionNeeded, ReaderResult

# ---------------------------------------------------------------------------
# Label-matching patterns
# ---------------------------------------------------------------------------

GENE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^ENS[A-Z]*G\d+"),
    re.compile(r"^[A-Z][A-Z0-9\-]{1,15}$"),
    re.compile(r"^[A-Z][a-z0-9\-]{1,15}$"),
]

BARCODE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^[ACGT]{8,}"),
    re.compile(r"^[A-Za-z0-9]+[-_]\d+$"),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _fraction_matching(labels: List[str], patterns: List[re.Pattern]) -> float:
    """Return the fraction of *labels* matching any of the given regex *patterns*."""
    if not labels:
        return 0.0
    count = sum(
        1 for lab in labels if any(pat.search(lab) for pat in patterns)
    )
    return count / len(labels)


def _infer_orientation(
    row_labels: List[str], col_labels: List[str], shape: tuple
) -> str:
    """Decide whether the matrix is cells-x-genes, genes-x-cells, or ambiguous.

    Returns one of ``"cells_x_genes"``, ``"genes_x_cells"``, or ``"ambiguous"``.
    """
    rows_gene_like = _fraction_matching(row_labels, GENE_PATTERNS)
    cols_gene_like = _fraction_matching(col_labels, GENE_PATTERNS)
    rows_barcode_like = _fraction_matching(row_labels, BARCODE_PATTERNS)
    cols_barcode_like = _fraction_matching(col_labels, BARCODE_PATTERNS)

    n_rows, n_cols = shape

    # Heuristic 1
    if rows_gene_like > 0.5 and cols_barcode_like > 0.3:
        return "genes_x_cells"
    # Heuristic 2
    if cols_gene_like > 0.5 and rows_barcode_like > 0.3:
        return "cells_x_genes"
    # Heuristic 3
    if rows_gene_like > 0.5 and cols_gene_like < 0.2:
        return "genes_x_cells"
    # Heuristic 4
    if cols_gene_like > 0.5 and rows_gene_like < 0.2:
        return "cells_x_genes"
    # Heuristic 5
    if n_cols > 10000 and n_rows < n_cols:
        return "cells_x_genes"
    # Heuristic 6
    if n_rows > 10000 and n_cols < n_rows:
        return "genes_x_cells"

    return "ambiguous"


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


def read_csv(
    path: str, delimiter: Optional[str] = None, decisions: Optional[dict] = None
) -> ReaderResult:
    """Read a CSV or TSV expression matrix.

    Parameters
    ----------
    path : str
        Path to the CSV/TSV file (may be gzip-compressed).
    delimiter : str, optional
        Column delimiter.  Inferred from extension when *None*:
        ``.tsv`` / ``.txt`` -> ``"\\t"``, otherwise ``","``.
    decisions : dict, optional
        Pre-resolved decisions.  Recognised key:
        ``"matrix_orientation"`` with value ``"cells_x_genes"`` or
        ``"genes_x_cells"``.

    Returns
    -------
    ReaderResult
    """
    decisions = decisions or {}
    warnings_list: list[str] = []

    # 1. Infer delimiter from extension (strip .gz first) -------------------
    basename = os.path.basename(path)
    decompressed = basename.endswith(".gz")
    ext_name = basename[:-3] if decompressed else basename  # strip .gz
    _, ext = os.path.splitext(ext_name)
    ext = ext.lower()

    if delimiter is None:
        if ext in (".tsv", ".txt"):
            delimiter = "\t"
        else:
            delimiter = ","

    source_format = "tsv" if delimiter == "\t" else "csv"

    # 2. Read table ---------------------------------------------------------
    df = pd.read_csv(path, sep=delimiter, index_col=0)

    # 3. Label arrays -------------------------------------------------------
    row_labels = [str(x) for x in df.index]
    col_labels = [str(x) for x in df.columns]

    # 4. Determine orientation ----------------------------------------------
    orientation = decisions.get("matrix_orientation")
    if orientation is None:
        orientation = _infer_orientation(row_labels, col_labels, df.shape)

    transposed = False

    # 5. Handle orientation -------------------------------------------------
    if orientation == "cells_x_genes":
        pass  # keep as-is
    elif orientation == "genes_x_cells":
        df = df.T
        transposed = True
        row_labels = [str(x) for x in df.index]
        col_labels = [str(x) for x in df.columns]
    elif orientation == "ambiguous":
        sample_rows = row_labels[:5]
        sample_cols = col_labels[:5]
        context = (
            f"Cannot determine matrix orientation for shape {df.shape}. "
            f"Sample row labels: {sample_rows}. "
            f"Sample column labels: {sample_cols}."
        )
        raise DecisionNeeded(
            decision_type="matrix_orientation",
            context=context,
            options=["cells_x_genes", "genes_x_cells"],
        )

    # 6. Convert to sparse csr_matrix (float32) -----------------------------
    X = sparse.csr_matrix(df.values.astype(np.float32))

    # 7. Build AnnData -------------------------------------------------------
    obs = pd.DataFrame(index=row_labels)
    var = pd.DataFrame(index=col_labels)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # 8. Source metadata -----------------------------------------------------
    if transposed:
        matrix_orientation_before = "genes_x_cells"
    elif orientation and orientation != "ambiguous":
        matrix_orientation_before = orientation
    else:
        matrix_orientation_before = "cells_x_genes"

    source_meta = {
        "source_format": source_format,
        "reader_used": "csv_reader",
        "matrix_orientation_before": matrix_orientation_before,
        "transposed": transposed,
        "raw_counts_found": False,
        "feature_types_present": [],
        "matrix_type_hint": None,
        "decompressed": decompressed,
        "warnings": warnings_list,
    }

    return ReaderResult(adata=adata, source_meta=source_meta)
