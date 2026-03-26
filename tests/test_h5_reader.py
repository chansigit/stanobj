"""Tests for scripts.readers.h5_reader."""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.base import DecisionNeeded
from scripts.readers.h5_reader import read_10x_h5, read_generic_h5


# ---------------------------------------------------------------------------
# 10x HDF5 reader
# ---------------------------------------------------------------------------


class TestRead10xH5:
    def test_basic_read(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        adata = result.adata
        # 10x h5 stores genes x cells; after transpose -> (10 cells, 20 genes)
        assert adata.shape == (10, 20)
        assert result.source_meta["source_format"] == "10x_h5"
        assert result.source_meta["transposed"] is True

    def test_gene_metadata(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        adata = result.adata
        assert "gene_id" in adata.var.columns
        assert "gene_symbol" in adata.var.columns
        assert "feature_type" in adata.var.columns

    def test_reader_used(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        assert result.source_meta["reader_used"] == "h5_reader:10x"

    def test_raw_counts_found(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        assert result.source_meta["raw_counts_found"] is True
        assert result.source_meta["matrix_type_hint"] == "counts"

    def test_feature_types_present(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        assert "feature_types_present" in result.source_meta
        assert "Gene Expression" in result.source_meta["feature_types_present"]

    def test_barcodes_as_obs_index(self, tiny_10x_h5_path):
        result = read_10x_h5(tiny_10x_h5_path)
        adata = result.adata
        assert any("AAACCTGA" in name for name in adata.obs_names)


# ---------------------------------------------------------------------------
# Generic HDF5 reader
# ---------------------------------------------------------------------------


class TestReadGenericH5:
    def test_unrecognized_raises_decision(self, tmp_dir):
        """An HDF5 with random structure should raise DecisionNeeded."""
        path = os.path.join(tmp_dir, "mystery.h5")
        with h5py.File(path, "w") as f:
            f.create_group("some_group")
            f.create_dataset("random_data", data=np.zeros((5, 5)))
            grp = f.create_group("nested")
            grp.create_dataset("values", data=np.ones(10))

        with pytest.raises(DecisionNeeded) as exc_info:
            read_generic_h5(path)

        assert exc_info.value.decision_type == "format_detection"
        assert "10x_h5" in exc_info.value.options
        assert "h5ad" in exc_info.value.options
        assert "skip" in exc_info.value.options
