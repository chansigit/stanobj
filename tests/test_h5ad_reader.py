"""Tests for scripts.readers.h5ad_reader."""

from __future__ import annotations

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.h5ad_reader import read_h5ad


# ---------------------------------------------------------------------------
# h5ad reader
# ---------------------------------------------------------------------------


class TestReadH5ad:
    def test_basic_read(self, tiny_h5ad_path):
        result = read_h5ad(tiny_h5ad_path)
        adata = result.adata
        assert adata.shape == (10, 20)
        assert result.source_meta["source_format"] == "h5ad"

    def test_reader_used(self, tiny_h5ad_path):
        result = read_h5ad(tiny_h5ad_path)
        assert result.source_meta["reader_used"] == "h5ad_reader"

    def test_not_transposed(self, tiny_h5ad_path):
        result = read_h5ad(tiny_h5ad_path)
        assert result.source_meta["transposed"] is False
        assert result.source_meta["matrix_orientation_before"] == "cells_x_genes"

    def test_preserves_layers(self, tmp_dir):
        """h5ad with layers={"counts": X} should report available_layers."""
        rng = np.random.default_rng(42)
        X = sparse.csr_matrix(rng.poisson(2, size=(10, 20)).astype(np.float32))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
            layers={"counts": X.copy()},
        )
        path = os.path.join(tmp_dir, "with_layers.h5ad")
        adata.write_h5ad(path)

        result = read_h5ad(path)
        assert "available_layers" in result.source_meta
        assert "counts" in result.source_meta["available_layers"]

    def test_preserves_obsm(self, tmp_dir):
        """h5ad with X_pca and X_umap in obsm should preserve them."""
        rng = np.random.default_rng(42)
        X = sparse.csr_matrix(rng.poisson(2, size=(10, 20)).astype(np.float32))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
        )
        adata.obsm["X_pca"] = rng.standard_normal((10, 50)).astype(np.float32)
        adata.obsm["X_umap"] = rng.standard_normal((10, 2)).astype(np.float32)
        path = os.path.join(tmp_dir, "with_obsm.h5ad")
        adata.write_h5ad(path)

        result = read_h5ad(path)
        assert "X_pca" in result.adata.obsm
        assert "X_umap" in result.adata.obsm
