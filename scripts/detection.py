"""Format detection and matrix type classification for stanobj."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

try:
    from .utils import decompress_to_temp, is_integer_like, strip_compression_ext
except ImportError:
    from utils import decompress_to_temp, is_integer_like, strip_compression_ext

# ---------------------------------------------------------------------------
# Extension map (after stripping compression)
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".h5ad": "h5ad",
    ".rds": "seurat_rds",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".loom": "loom",
    ".mtx": "mtx",
}

# Extensions that require HDF5 inspection
_HDF5_EXTS = {".h5", ".hdf5"}

# ---------------------------------------------------------------------------
# Source hint → canonical matrix type
# ---------------------------------------------------------------------------

_HINT_MAP = {
    "counts": "counts",
    "raw_counts": "counts",
    "data": "log1p",
    "logcounts": "log1p",
    "normcounts": "normalized",
    "scale.data": "scaled",
}


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def detect_format(path: str) -> str:
    """Auto-detect input format from file extension, directory contents,
    and HDF5 internal structure.

    Parameters
    ----------
    path
        Path to a file or directory.

    Returns
    -------
    str
        One of: ``"h5ad"``, ``"seurat_rds"``, ``"csv"``, ``"tsv"``,
        ``"loom"``, ``"mtx"``, ``"10x_h5"``, ``"generic_h5"``.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    path = str(path)

    # --- Directory: check for matrix.mtx* ---
    if os.path.isdir(path):
        for entry in os.listdir(path):
            if entry.startswith("matrix.mtx"):
                return "mtx"
        raise ValueError(
            f"Cannot detect format: directory '{path}' does not contain matrix.mtx*"
        )

    # --- File: strip compression, then map extension ---
    stripped, was_compressed = strip_compression_ext(path)
    ext = os.path.splitext(stripped)[1].lower()

    # Direct extension lookup
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    # HDF5 files need internal inspection
    if ext in _HDF5_EXTS:
        return _inspect_hdf5(path, was_compressed)

    raise ValueError(f"Cannot detect format for path: '{path}'")


def _inspect_hdf5(path: str, is_compressed: bool) -> str:
    """Open an HDF5 file and determine whether it is 10x, h5ad, or generic.

    If the file is gzip/bz2-compressed, decompress to a temporary file first.
    """
    import h5py

    tmp_path: str | None = None
    try:
        if is_compressed:
            tmp_path = decompress_to_temp(path)
            inspect_path = tmp_path
        else:
            inspect_path = path

        with h5py.File(inspect_path, "r") as f:
            if "matrix" in f and isinstance(f["matrix"], h5py.Group):
                return "10x_h5"
            if "X" in f or "obs" in f:
                return "h5ad"
            return "generic_h5"
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# compute_evidence
# ---------------------------------------------------------------------------


def compute_evidence(X) -> dict:
    """Compute evidence metrics from a matrix for type classification.

    Parameters
    ----------
    X
        A dense ndarray or scipy sparse matrix (cells x genes).

    Returns
    -------
    dict
        Keys: has_negatives, is_integer, min_value, max_value, mean_value,
        sparsity, row_sums_cv, near_zero_mean_frac, near_one_std_frac,
        n_cells_sampled, n_genes.
    """
    # Convert to dense if sparse
    if sparse.issparse(X):
        Xd = X.toarray()
    else:
        Xd = np.asarray(X, dtype=float)

    n_cells, n_genes = Xd.shape

    # Replace NaN/Inf with 0 for statistics; otherwise np.min/max/mean
    # return NaN and corrupt both the classifier and the JSON report.
    if not np.isfinite(Xd).all():
        Xd = np.where(np.isfinite(Xd), Xd, 0.0)

    has_negatives = bool(np.any(Xd < 0))
    is_int = is_integer_like(Xd)
    min_val = float(np.min(Xd))
    max_val = float(np.max(Xd))
    mean_val = float(np.mean(Xd))

    # Sparsity: fraction of zeros
    n_zeros = np.sum(Xd == 0)
    sparsity = float(n_zeros / Xd.size) if Xd.size > 0 else 0.0

    # Row sums CV (coefficient of variation)
    row_sums = Xd.sum(axis=1)
    row_mean = row_sums.mean()
    if row_mean != 0:
        row_sums_cv = float(row_sums.std() / row_mean)
    else:
        row_sums_cv = 0.0

    # Per-column (gene) statistics
    col_means = Xd.mean(axis=0)
    col_stds = Xd.std(axis=0)

    # Fraction of columns with mean near zero (|mean| < 0.5)
    near_zero_mean_frac = float(np.mean(np.abs(col_means) < 0.5))

    # Fraction of columns with std near one (|std - 1| < 0.5)
    near_one_std_frac = float(np.mean(np.abs(col_stds - 1.0) < 0.5))

    # Round all float values to 4 decimal places
    return {
        "has_negatives": has_negatives,
        "is_integer": is_int,
        "min_value": round(min_val, 4),
        "max_value": round(max_val, 4),
        "mean_value": round(mean_val, 4),
        "sparsity": round(sparsity, 4),
        "row_sums_cv": round(row_sums_cv, 4),
        "near_zero_mean_frac": round(near_zero_mean_frac, 4),
        "near_one_std_frac": round(near_one_std_frac, 4),
        "n_cells_sampled": n_cells,
        "n_genes": n_genes,
    }


# ---------------------------------------------------------------------------
# classify_matrix_type
# ---------------------------------------------------------------------------


def classify_matrix_type(evidence: dict) -> Tuple[str, str]:
    """Decision-tree classifier for matrix type based on evidence metrics.

    Parameters
    ----------
    evidence
        Dict produced by :func:`compute_evidence`.

    Returns
    -------
    tuple[str, str]
        ``(matrix_type, confidence)`` where *matrix_type* is one of
        ``"counts"``, ``"log1p"``, ``"normalized"``, ``"scaled"``,
        ``"unknown"``; and *confidence* is ``"high"``, ``"medium"``, or
        ``"low"``.
    """
    has_neg = evidence["has_negatives"]
    is_int = evidence["is_integer"]
    max_val = evidence["max_value"]
    near_zero = evidence["near_zero_mean_frac"]
    near_one = evidence["near_one_std_frac"]
    sparsity = evidence["sparsity"]
    row_cv = evidence["row_sums_cv"]

    # 1. Scaled: negative values + near-zero column means + near-one column stds
    if has_neg and near_zero > 0.7 and near_one > 0.7:
        return "scaled", "high"

    # 2. Counts: non-negative integer with reasonable max
    if not has_neg and is_int and max_val > 10:
        conf = "high" if sparsity > 0.8 else "medium"
        return "counts", conf

    # 3. Normalized: non-negative, non-integer, very uniform row sums, high max
    if not has_neg and not is_int and row_cv < 0.01 and max_val > 100:
        return "normalized", "high"

    # 4. Log1p: non-negative, non-integer, moderate max
    if not has_neg and not is_int and max_val < 20:
        conf = "high" if max_val < 15 else "medium"
        return "log1p", conf

    # 5. Normalized (low confidence): non-negative, non-integer, smallish row CV
    if not has_neg and not is_int and row_cv < 0.1:
        return "normalized", "low"

    # 6. Negative fallback → scaled (low confidence)
    if has_neg:
        return "scaled", "low"

    # 7. Catch-all
    return "unknown", "low"


# ---------------------------------------------------------------------------
# classify_with_override
# ---------------------------------------------------------------------------


def classify_with_override(
    evidence: dict, source_hint: Optional[str] = None
) -> dict:
    """Apply source metadata override on top of heuristic classification.

    Parameters
    ----------
    evidence
        Dict produced by :func:`compute_evidence`.
    source_hint
        Optional layer/slot name from the source object (e.g. ``"counts"``,
        ``"logcounts"``, ``"scale.data"``).

    Returns
    -------
    dict
        Keys: matrix_type, confidence, matrix_type_source, warnings.
    """
    heuristic_type, heuristic_conf = classify_matrix_type(evidence)
    warnings: list[str] = []

    if source_hint is None or source_hint not in _HINT_MAP:
        return {
            "matrix_type": heuristic_type,
            "confidence": heuristic_conf,
            "matrix_type_source": "heuristic",
            "warnings": warnings,
        }

    hint_type = _HINT_MAP[source_hint]

    if hint_type == heuristic_type:
        # Agreement: high confidence, sourced from metadata
        return {
            "matrix_type": hint_type,
            "confidence": "high",
            "matrix_type_source": "source_metadata",
            "warnings": warnings,
        }

    # Mismatch: trust source but warn
    warnings.append(
        f"Source metadata suggests '{hint_type}' (from hint '{source_hint}') "
        f"but heuristic detected '{heuristic_type}'. Trusting source metadata."
    )
    return {
        "matrix_type": hint_type,
        "confidence": "medium",
        "matrix_type_source": "source_metadata",
        "warnings": warnings,
    }
