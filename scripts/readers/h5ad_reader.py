"""Reader for h5ad (AnnData HDF5) files."""

from __future__ import annotations

from typing import Optional

import anndata as ad

try:
    from .base import ReaderResult
except ImportError:
    from base import ReaderResult


def read_h5ad(path: str, decisions: Optional[dict] = None) -> ReaderResult:
    """Read an ``.h5ad`` file using :func:`anndata.read_h5ad`.

    The AnnData object is returned as-is (full load, no backed mode).
    Downstream standardisation will handle normalisation of obs/var/layers.

    Parameters
    ----------
    path : str
        Path to the ``.h5ad`` file.
    decisions : dict, optional
        Reserved for future decision-based interactivity.

    Returns
    -------
    ReaderResult
    """
    adata = ad.read_h5ad(path)

    # Collect available layer names
    available_layers = list(adata.layers.keys())

    source_meta = {
        "source_format": "h5ad",
        "reader_used": "h5ad_reader",
        "matrix_orientation_before": "cells_x_genes",
        "transposed": False,
        "available_layers": available_layers,
    }

    return ReaderResult(adata=adata, source_meta=source_meta)
