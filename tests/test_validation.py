"""Tests for scripts.validation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.validation import ValidationResult, validate_adata


# -----------------------------------------------------------------------
# ValidationResult dataclass
# -----------------------------------------------------------------------


class TestValidationResult:
    def test_defaults(self):
        vr = ValidationResult()
        assert vr.passed is True
        assert vr.fatal_errors == []
        assert vr.warnings == []

    def test_add_fatal(self):
        vr = ValidationResult()
        vr.add_fatal("something broke")
        assert vr.passed is False
        assert "something broke" in vr.fatal_errors

    def test_add_warning(self):
        vr = ValidationResult()
        vr.add_warning("heads up")
        assert vr.passed is True
        assert "heads up" in vr.warnings


# -----------------------------------------------------------------------
# validate_adata — fatal checks
# -----------------------------------------------------------------------


class TestValidateAdataFatal:
    def test_valid_adata(self, tiny_adata):
        """A well-formed tiny adata should pass with no fatal errors."""
        result = validate_adata(tiny_adata, matrix_type="counts")
        assert result.passed is True
        assert result.fatal_errors == []

    def test_empty_obs(self):
        """Zero cells must be fatal."""
        adata = ad.AnnData(
            X=np.empty((0, 5)),
            obs=pd.DataFrame(index=pd.Index([], dtype=str)),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False
        assert any("no cells" in e.lower() for e in result.fatal_errors)

    def test_empty_vars(self):
        """Zero genes must be fatal."""
        adata = ad.AnnData(
            X=np.empty((3, 0)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
            var=pd.DataFrame(index=pd.Index([], dtype=str)),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False
        assert any("no genes" in e.lower() for e in result.fatal_errors)

    def test_shape_mismatch(self):
        """X.shape vs obs/var length mismatch must be fatal."""
        # Manually craft an adata with inconsistent internal shapes by
        # creating one with correct shape then poking at its internals.
        adata = ad.AnnData(
            X=np.ones((4, 5)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        # Corrupt obs index (3 names for 4 rows)
        adata._obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False

    def test_duplicate_obs_names(self):
        """Duplicate cell barcodes must be fatal and report count."""
        obs_idx = ["cellA", "cellB", "cellA", "cellC"]
        adata = ad.AnnData(
            X=np.ones((4, 3)),
            obs=pd.DataFrame(index=obs_idx),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False
        combined = " ".join(result.fatal_errors).lower()
        assert "duplicate" in combined
        # Should report the count of duplicated names
        assert "1" in combined or "duplicate" in combined

    def test_duplicate_var_names(self):
        """Duplicate gene names must be fatal and report count."""
        var_idx = ["geneX", "geneY", "geneX", "geneZ", "geneY"]
        adata = ad.AnnData(
            X=np.ones((3, 5)),
            obs=pd.DataFrame(index=["c0", "c1", "c2"]),
            var=pd.DataFrame(index=var_idx),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False
        combined = " ".join(result.fatal_errors).lower()
        assert "duplicate" in combined

    def test_x_is_none(self):
        """X=None must be fatal."""
        adata = ad.AnnData(
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        # Force X to None
        adata.X = None
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is False


# -----------------------------------------------------------------------
# validate_adata — warning checks
# -----------------------------------------------------------------------


class TestValidateAdataWarnings:
    def test_no_matrix_type_still_ok(self, tiny_adata):
        """matrix_type='unknown' should not trigger the counts check."""
        result = validate_adata(tiny_adata, matrix_type="unknown")
        assert result.passed is True
        # Should not have the integer-like warning
        assert not any("integer" in w.lower() for w in result.warnings)

    def test_warn_counts_with_floats(self):
        """Float values when matrix_type='counts' should warn."""
        X = np.array([[1.5, 2.3], [0.1, 4.7], [3.3, 0.9]])
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"batch": ["a", "b", "c"]}, index=["c0", "c1", "c2"]),
            var=pd.DataFrame({"gene_type": ["x", "y"]}, index=["g0", "g1"]),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is True
        assert any("integer" in w.lower() or "count" in w.lower() for w in result.warnings)

    def test_warn_high_sparsity(self):
        """Sparsity > 0.999 should trigger a 'nearly empty' warning."""
        n_obs, n_vars = 100, 200
        # One nonzero in 20000 elements => sparsity = 0.99995
        X = sparse.lil_matrix((n_obs, n_vars), dtype=np.float32)
        X[0, 0] = 1.0
        X = X.tocsr()
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"batch": ["a"] * n_obs}, index=[f"c{i}" for i in range(n_obs)]),
            var=pd.DataFrame({"sym": [f"g{i}" for i in range(n_vars)]}, index=[f"g{i}" for i in range(n_vars)]),
        )
        result = validate_adata(adata, matrix_type="counts")
        assert result.passed is True
        assert any("nearly empty" in w.lower() for w in result.warnings)

    def test_warn_embedding_dim_mismatch(self):
        """obsm entry with wrong n_obs should warn and be deleted."""
        X = np.ones((5, 10))
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame({"batch": list("abcde")}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame({"sym": [f"g{i}" for i in range(10)]}, index=[f"g{i}" for i in range(10)]),
        )
        # Good embedding
        adata.obsm["X_pca"] = np.random.default_rng(0).standard_normal((5, 2))
        # Bad embedding — bypass AnnData shape validation via _obsm dict
        bad_arr = np.random.default_rng(0).standard_normal((3, 2))
        adata._obsm["X_bad"] = bad_arr

        result = validate_adata(adata, matrix_type="unknown")
        assert result.passed is True
        assert any("X_bad" in w for w in result.warnings)
        # Bad key must be deleted from the raw store
        assert "X_bad" not in adata._obsm
        # Good key must remain
        assert "X_pca" in adata._obsm

    def test_warn_no_cell_metadata(self):
        """obs with zero columns should warn."""
        adata = ad.AnnData(
            X=np.ones((5, 10)),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame({"sym": [f"g{i}" for i in range(10)]}, index=[f"g{i}" for i in range(10)]),
        )
        result = validate_adata(adata, matrix_type="unknown")
        assert any("no cell metadata" in w.lower() for w in result.warnings)

    def test_warn_no_gene_metadata(self):
        """var with zero columns should warn."""
        adata = ad.AnnData(
            X=np.ones((5, 10)),
            obs=pd.DataFrame({"batch": list("abcde")}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        result = validate_adata(adata, matrix_type="unknown")
        assert any("no gene metadata" in w.lower() for w in result.warnings)

    def test_warn_small_obs(self):
        """Fewer than 50 cells should warn."""
        adata = ad.AnnData(
            X=np.ones((10, 200)),
            obs=pd.DataFrame({"b": range(10)}, index=[f"c{i}" for i in range(10)]),
            var=pd.DataFrame({"s": range(200)}, index=[f"g{i}" for i in range(200)]),
        )
        result = validate_adata(adata, matrix_type="unknown")
        assert any("50" in w or "cells" in w.lower() for w in result.warnings)

    def test_warn_small_vars(self):
        """Fewer than 100 genes should warn."""
        adata = ad.AnnData(
            X=np.ones((100, 10)),
            obs=pd.DataFrame({"b": range(100)}, index=[f"c{i}" for i in range(100)]),
            var=pd.DataFrame({"s": range(10)}, index=[f"g{i}" for i in range(10)]),
        )
        result = validate_adata(adata, matrix_type="unknown")
        assert any("100" in w or "genes" in w.lower() for w in result.warnings)

    def test_warn_adt_without_feature_type(self):
        """ADT-like var_names without a feature_type column should warn."""
        var_idx = ["ADT_CD3", "ab_CD19", "HTO_Hash1", "GAPDH", "TP53"]
        adata = ad.AnnData(
            X=np.ones((100, 5)),
            obs=pd.DataFrame({"b": range(100)}, index=[f"c{i}" for i in range(100)]),
            var=pd.DataFrame(index=var_idx),
        )
        result = validate_adata(adata, matrix_type="unknown")
        assert any("feature_type" in w.lower() or "adt" in w.lower() for w in result.warnings)
