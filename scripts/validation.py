"""Validation checks for AnnData objects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import sparse

from scripts.utils import is_integer_like, sample_matrix


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_adata`.

    Attributes
    ----------
    passed : bool
        ``False`` if any fatal error was recorded.
    fatal_errors : list[str]
        Messages for checks that block further processing.
    warnings : list[str]
        Non-blocking advisory messages.
    """

    passed: bool = True
    fatal_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_fatal(self, msg: str) -> None:
        """Register a fatal validation error (sets *passed* to ``False``)."""
        self.passed = False
        self.fatal_errors.append(msg)

    def add_warning(self, msg: str) -> None:
        """Register a non-blocking warning."""
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# ADT pattern for heuristic feature-type detection
# ---------------------------------------------------------------------------

_ADT_PREFIX_RE = re.compile(r"^(ADT_|HTO_|ab_)", re.IGNORECASE)
_ADT_SUFFIX_RE = re.compile(r"_TotalSeq", re.IGNORECASE)


def _looks_like_adt(name: str) -> bool:
    """Return True if *name* looks like an antibody-derived tag."""
    return bool(_ADT_PREFIX_RE.match(name) or _ADT_SUFFIX_RE.search(name))


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_adata(
    adata: "anndata.AnnData",  # noqa: F821
    matrix_type: str,
) -> ValidationResult:
    """Run fatal and warning checks on *adata*.

    Parameters
    ----------
    adata
        The AnnData to validate.
    matrix_type
        Declared matrix type (e.g. ``"counts"``, ``"normalized"``,
        ``"unknown"``).  Only ``"counts"`` triggers the integer-check
        warning.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # ------------------------------------------------------------------
    # Fatal checks
    # ------------------------------------------------------------------

    # X present?
    if adata.X is None:
        result.add_fatal("X matrix is None")
        return result

    # X is 2-D?
    if adata.X.ndim != 2:
        result.add_fatal(f"X is not 2-D (ndim={adata.X.ndim})")
        return result

    n_obs, n_vars = adata.X.shape

    # Non-empty dimensions
    if n_obs == 0:
        result.add_fatal("No cells (n_obs == 0)")
    if n_vars == 0:
        result.add_fatal("No genes (n_vars == 0)")

    # Shape vs index consistency
    if adata.X.shape[0] != len(adata.obs_names):
        result.add_fatal(
            f"X.shape[0] ({adata.X.shape[0]}) != len(obs_names) "
            f"({len(adata.obs_names)})"
        )
    if adata.X.shape[1] != len(adata.var_names):
        result.add_fatal(
            f"X.shape[1] ({adata.X.shape[1]}) != len(var_names) "
            f"({len(adata.var_names)})"
        )

    # Uniqueness of names
    obs_names = adata.obs_names
    if not obs_names.is_unique:
        n_dup = int(obs_names.duplicated().sum())
        result.add_fatal(
            f"obs_names are not unique ({n_dup} duplicate name(s))"
        )

    var_names = adata.var_names
    if not var_names.is_unique:
        n_dup = int(var_names.duplicated().sum())
        result.add_fatal(
            f"var_names are not unique ({n_dup} duplicate name(s))"
        )

    # Early return if any fatal check fired
    if not result.passed:
        return result

    # ------------------------------------------------------------------
    # Warning checks (only reached when no fatal errors)
    # ------------------------------------------------------------------

    # Count-matrix integrity (sample up to 200 cells)
    if matrix_type == "counts":
        sample = sample_matrix(adata.X, n_sample=200)
        if not is_integer_like(sample):
            result.add_warning(
                "matrix_type is 'counts' but values are not integer-like"
            )
        elif np.any(sample < 0):
            result.add_warning(
                "matrix_type is 'counts' but matrix contains negative values"
            )

    # Missing metadata columns
    if len(adata.obs.columns) == 0:
        result.add_warning("No cell metadata columns in obs")
    if len(adata.var.columns) == 0:
        result.add_warning("No gene metadata columns in var")

    # Sparsity
    if sparse.issparse(adata.X):
        nnz = adata.X.nnz
    else:
        nnz = int(np.count_nonzero(adata.X))
    total = n_obs * n_vars
    if total > 0:
        sparsity = 1.0 - nnz / total
        if sparsity > 0.999:
            result.add_warning(
                f"Matrix is nearly empty (sparsity {sparsity:.4%})"
            )

    # Small dataset hints
    if n_obs < 50:
        result.add_warning(f"Very few cells ({n_obs} < 50)")
    if n_vars < 100:
        result.add_warning(f"Very few genes ({n_vars} < 100)")

    # obsm shape consistency — remove bad entries
    # Access the raw _obsm dict to avoid AnnData's own shape validation
    # which would raise before we can inspect/clean entries.
    obsm_store = adata._obsm
    bad_obsm_keys: list[str] = []
    for key in list(obsm_store.keys()):
        entry = obsm_store[key]
        if hasattr(entry, "shape") and entry.shape[0] != n_obs:
            result.add_warning(
                f"obsm['{key}'] has {entry.shape[0]} rows, expected {n_obs}; "
                f"removing it"
            )
            bad_obsm_keys.append(key)
    for key in bad_obsm_keys:
        del obsm_store[key]

    # ADT heuristic: warn if var_names look like ADT but no feature_type col
    if "feature_type" not in adata.var.columns:
        adt_count = sum(1 for name in adata.var_names if _looks_like_adt(name))
        if adt_count > 0:
            result.add_warning(
                f"No 'feature_type' column in var but {adt_count} var_name(s) "
                f"look like ADT/HTO markers"
            )

    return result
