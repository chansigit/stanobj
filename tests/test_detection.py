"""Tests for scripts.detection — format detection and matrix type classification."""

from __future__ import annotations

import gzip
import os
import sys

import h5py
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from detection import (
    classify_matrix_type,
    classify_with_override,
    compute_evidence,
    detect_format,
)


# -----------------------------------------------------------------------
# detect_format
# -----------------------------------------------------------------------


class TestDetectFormat:
    """Test auto-detection of input formats from path / extension."""

    def test_h5ad(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.h5ad")
        open(path, "w").close()
        assert detect_format(path) == "h5ad"

    def test_rds(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.rds")
        open(path, "w").close()
        assert detect_format(path) == "seurat_rds"

    def test_csv(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.csv")
        open(path, "w").close()
        assert detect_format(path) == "csv"

    def test_tsv(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.tsv")
        open(path, "w").close()
        assert detect_format(path) == "tsv"

    def test_txt(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.txt")
        open(path, "w").close()
        assert detect_format(path) == "tsv"

    def test_csv_gz(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.csv.gz")
        with gzip.open(path, "wb") as f:
            f.write(b"a,b\n1,2\n")
        assert detect_format(path) == "csv"

    def test_h5_10x(self, tiny_10x_h5_path):
        assert detect_format(tiny_10x_h5_path) == "10x_h5"

    def test_h5_h5ad_like(self, tmp_dir):
        """An .h5 file with X/ and obs/ groups should be detected as h5ad."""
        path = os.path.join(tmp_dir, "adata.h5")
        with h5py.File(path, "w") as f:
            f.create_group("X")
            f.create_group("obs")
        assert detect_format(path) == "h5ad"

    def test_h5_generic(self, tmp_dir):
        """An .h5 file with no recognized structure returns generic_h5."""
        path = os.path.join(tmp_dir, "custom.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("values", data=[1, 2, 3])
        assert detect_format(path) == "generic_h5"

    def test_hdf5_extension(self, tmp_dir):
        """A .hdf5 file should also be inspected like .h5."""
        path = os.path.join(tmp_dir, "custom.hdf5")
        with h5py.File(path, "w") as f:
            f.create_group("matrix")
        assert detect_format(path) == "10x_h5"

    def test_mtx_dir(self, tiny_mtx_dir):
        assert detect_format(tiny_mtx_dir) == "mtx"

    def test_mtx_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "matrix.mtx")
        open(path, "w").close()
        assert detect_format(path) == "mtx"

    def test_loom(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.loom")
        open(path, "w").close()
        assert detect_format(path) == "loom"

    def test_unknown_raises(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.xyz")
        open(path, "w").close()
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(path)


# -----------------------------------------------------------------------
# compute_evidence
# -----------------------------------------------------------------------


class TestComputeEvidence:
    """Test evidence computation from count / scaled matrices."""

    def test_count_matrix(self):
        """Poisson-distributed counts should be integer with high sparsity."""
        rng = np.random.default_rng(42)
        X = rng.poisson(lam=2, size=(100, 50)).astype(np.float64)
        mask = rng.random((100, 50)) < 0.6
        X[mask] = 0

        ev = compute_evidence(X)
        assert ev["is_integer"] is True
        assert ev["has_negatives"] is False
        assert ev["min_value"] >= 0
        assert ev["sparsity"] > 0.5
        assert "row_sums_cv" in ev
        assert "near_zero_mean_frac" in ev
        assert "near_one_std_frac" in ev
        assert ev["n_cells_sampled"] == 100
        assert ev["n_genes"] == 50

    def test_scaled_matrix(self):
        """Standard-normal matrix should have negatives, near-zero mean columns."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((200, 50))

        ev = compute_evidence(X)
        assert ev["has_negatives"] is True
        assert ev["is_integer"] is False
        assert ev["near_zero_mean_frac"] > 0.5
        assert ev["near_one_std_frac"] > 0.3
        assert ev["n_cells_sampled"] == 200
        assert ev["n_genes"] == 50

    def test_sparse_input(self):
        """Sparse matrices should work the same as dense."""
        rng = np.random.default_rng(0)
        dense = rng.poisson(lam=1, size=(50, 30)).astype(np.float64)
        sp = sparse.csr_matrix(dense)

        ev = compute_evidence(sp)
        assert ev["is_integer"] is True
        assert ev["has_negatives"] is False
        assert ev["n_cells_sampled"] == 50
        assert ev["n_genes"] == 30

    def test_values_rounded(self):
        """Numeric evidence values should be rounded to 4 decimal places."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 20))
        ev = compute_evidence(X)

        for key in ("min_value", "max_value", "mean_value", "sparsity",
                     "row_sums_cv", "near_zero_mean_frac", "near_one_std_frac"):
            val = ev[key]
            if isinstance(val, float):
                assert val == round(val, 4), f"{key} not rounded: {val}"


# -----------------------------------------------------------------------
# classify_matrix_type
# -----------------------------------------------------------------------


class TestClassifyMatrixType:
    """Test the decision-tree classifier with curated evidence dicts."""

    def test_counts(self):
        evidence = dict(
            has_negatives=False,
            is_integer=True,
            min_value=0,
            max_value=5000,
            mean_value=2.5,
            sparsity=0.85,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "counts"
        assert conf == "high"

    def test_counts_medium_confidence(self):
        evidence = dict(
            has_negatives=False,
            is_integer=True,
            min_value=0,
            max_value=500,
            mean_value=2.0,
            sparsity=0.5,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "counts"
        assert conf == "medium"

    def test_log1p(self):
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=17.0,
            mean_value=1.5,
            sparsity=0.6,
            row_sums_cv=0.5,
            near_zero_mean_frac=0.2,
            near_one_std_frac=0.2,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "log1p"
        assert conf == "medium"

    def test_log1p_high_confidence(self):
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=8.0,
            mean_value=1.5,
            sparsity=0.6,
            row_sums_cv=0.5,
            near_zero_mean_frac=0.2,
            near_one_std_frac=0.2,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "log1p"
        assert conf == "high"

    def test_scaled(self):
        evidence = dict(
            has_negatives=True,
            is_integer=False,
            min_value=-5.0,
            max_value=5.0,
            mean_value=0.01,
            sparsity=0.05,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.9,
            near_one_std_frac=0.85,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "scaled"
        assert conf == "high"

    def test_normalized(self):
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=500.0,
            mean_value=10.0,
            sparsity=0.6,
            row_sums_cv=0.005,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "normalized"
        assert conf == "high"

    def test_normalized_low(self):
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=500.0,
            mean_value=10.0,
            sparsity=0.6,
            row_sums_cv=0.05,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "normalized"
        assert conf == "low"

    def test_scaled_fallback(self):
        """Negative values but not clearly scaled → 'scaled', 'low'."""
        evidence = dict(
            has_negatives=True,
            is_integer=False,
            min_value=-2.0,
            max_value=10.0,
            mean_value=1.0,
            sparsity=0.3,
            row_sums_cv=0.5,
            near_zero_mean_frac=0.2,
            near_one_std_frac=0.2,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "scaled"
        assert conf == "low"

    def test_unknown(self):
        """Non-negative, non-integer, high max, high row_cv → 'unknown'."""
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=50000.0,
            mean_value=100.0,
            sparsity=0.3,
            row_sums_cv=0.5,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        mtype, conf = classify_matrix_type(evidence)
        assert mtype == "unknown"
        assert conf == "low"


# -----------------------------------------------------------------------
# classify_with_override
# -----------------------------------------------------------------------


class TestClassifyWithOverride:
    def test_override_wins_with_warning(self):
        """Source hint that disagrees with heuristic should still be trusted."""
        evidence = dict(
            has_negatives=False,
            is_integer=True,
            min_value=0,
            max_value=5000,
            mean_value=2.5,
            sparsity=0.85,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        # Heuristic would say "counts", but hint says "logcounts" -> "log1p"
        result = classify_with_override(evidence, source_hint="logcounts")
        assert result["matrix_type"] == "log1p"
        assert result["matrix_type_source"] == "source_metadata"
        assert result["confidence"] == "medium"
        assert len(result["warnings"]) > 0

    def test_override_matches(self):
        """Source hint that agrees with heuristic should boost confidence."""
        evidence = dict(
            has_negatives=False,
            is_integer=True,
            min_value=0,
            max_value=5000,
            mean_value=2.5,
            sparsity=0.85,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        result = classify_with_override(evidence, source_hint="counts")
        assert result["matrix_type"] == "counts"
        assert result["confidence"] == "high"
        assert result["matrix_type_source"] == "source_metadata"
        assert len(result["warnings"]) == 0

    def test_no_override(self):
        """No source hint should use pure heuristic."""
        evidence = dict(
            has_negatives=True,
            is_integer=False,
            min_value=-5.0,
            max_value=5.0,
            mean_value=0.01,
            sparsity=0.05,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.9,
            near_one_std_frac=0.85,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        result = classify_with_override(evidence)
        assert result["matrix_type"] == "scaled"
        assert result["matrix_type_source"] == "heuristic"
        assert result["confidence"] == "high"
        assert result["warnings"] == []

    def test_raw_counts_hint(self):
        """raw_counts hint should map to counts."""
        evidence = dict(
            has_negatives=False,
            is_integer=True,
            min_value=0,
            max_value=5000,
            mean_value=2.5,
            sparsity=0.85,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        result = classify_with_override(evidence, source_hint="raw_counts")
        assert result["matrix_type"] == "counts"
        assert result["confidence"] == "high"

    def test_scale_data_hint(self):
        """scale.data hint should map to scaled."""
        evidence = dict(
            has_negatives=True,
            is_integer=False,
            min_value=-5.0,
            max_value=5.0,
            mean_value=0.01,
            sparsity=0.05,
            row_sums_cv=0.3,
            near_zero_mean_frac=0.9,
            near_one_std_frac=0.85,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        result = classify_with_override(evidence, source_hint="scale.data")
        assert result["matrix_type"] == "scaled"
        assert result["confidence"] == "high"

    def test_normcounts_hint(self):
        """normcounts hint should map to normalized."""
        evidence = dict(
            has_negatives=False,
            is_integer=False,
            min_value=0,
            max_value=500.0,
            mean_value=10.0,
            sparsity=0.6,
            row_sums_cv=0.005,
            near_zero_mean_frac=0.1,
            near_one_std_frac=0.1,
            n_cells_sampled=1000,
            n_genes=2000,
        )
        result = classify_with_override(evidence, source_hint="normcounts")
        assert result["matrix_type"] == "normalized"
        assert result["confidence"] == "high"
