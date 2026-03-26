"""Tests for scripts.readers.loom_reader."""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.loom_reader import read_loom


# ---------------------------------------------------------------------------
# Fixture: tiny loom file
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_loom_path(tmp_dir):
    """Build a minimal loom HDF5 file.

    Layout (loom spec):
        /matrix          — (15, 8) float32 (genes x cells)
        /row_attrs/Gene  — 15 gene names as bytes
        /col_attrs/CellID      — 8 cell IDs as bytes
        /col_attrs/ClusterName — 8 cluster labels as bytes
    """
    path = os.path.join(tmp_dir, "tiny.loom")
    rng = np.random.default_rng(42)
    matrix = rng.poisson(lam=2, size=(15, 8)).astype(np.float32)

    gene_names = [f"Gene_{i}" for i in range(15)]
    cell_ids = [f"Cell_{j}" for j in range(8)]
    clusters = [f"Cluster_{j % 3}" for j in range(8)]

    with h5py.File(path, "w") as f:
        f.create_dataset("matrix", data=matrix)
        ra = f.create_group("row_attrs")
        ra.create_dataset("Gene", data=np.array(gene_names, dtype="S"))
        ca = f.create_group("col_attrs")
        ca.create_dataset("CellID", data=np.array(cell_ids, dtype="S"))
        ca.create_dataset("ClusterName", data=np.array(clusters, dtype="S"))

    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoomReader:
    def test_basic_read(self, tiny_loom_path):
        """Shape should be (8 cells, 15 genes) after transpose; transposed flag True."""
        result = read_loom(tiny_loom_path)
        adata = result.adata
        assert adata.shape == (8, 15)
        assert result.source_meta["transposed"] is True
        assert result.source_meta["source_format"] == "loom"
        assert result.source_meta["reader_used"] == "loom_reader"
        assert result.source_meta["matrix_orientation_before"] == "genes_x_cells"

    def test_metadata_extracted(self, tiny_loom_path):
        """ClusterName should appear in obs.columns; var should have 15 genes."""
        result = read_loom(tiny_loom_path)
        adata = result.adata
        assert "ClusterName" in adata.obs.columns
        assert adata.n_vars == 15

    def test_gene_names_as_var_index(self, tiny_loom_path):
        """Gene names from row_attrs/Gene should be used as var index."""
        result = read_loom(tiny_loom_path)
        adata = result.adata
        expected = [f"Gene_{i}" for i in range(15)]
        assert list(adata.var_names) == expected

    def test_cell_ids_as_obs_index(self, tiny_loom_path):
        """Cell IDs from col_attrs/CellID should be used as obs index."""
        result = read_loom(tiny_loom_path)
        adata = result.adata
        expected = [f"Cell_{j}" for j in range(8)]
        assert list(adata.obs_names) == expected

    def test_cluster_values(self, tiny_loom_path):
        """Cluster labels should round-trip correctly."""
        result = read_loom(tiny_loom_path)
        clusters = result.adata.obs["ClusterName"].tolist()
        expected = [f"Cluster_{j % 3}" for j in range(8)]
        assert clusters == expected

    def test_sparse_output(self, tiny_loom_path):
        """X should be stored as a sparse CSR matrix."""
        from scipy import sparse

        result = read_loom(tiny_loom_path)
        assert sparse.issparse(result.adata.X)

    def test_no_gene_key_fallback(self, tmp_dir):
        """When row_attrs has no recognised gene key, use integer indices."""
        path = os.path.join(tmp_dir, "no_gene_key.loom")
        rng = np.random.default_rng(7)
        matrix = rng.poisson(lam=1, size=(5, 3)).astype(np.float32)

        with h5py.File(path, "w") as f:
            f.create_dataset("matrix", data=matrix)
            ra = f.create_group("row_attrs")
            ra.create_dataset("Accession", data=np.array(["ACC_0", "ACC_1", "ACC_2", "ACC_3", "ACC_4"], dtype="S"))
            ca = f.create_group("col_attrs")
            ca.create_dataset("CellID", data=np.array(["C0", "C1", "C2"], dtype="S"))

        result = read_loom(path)
        adata = result.adata
        assert adata.shape == (3, 5)
        # var index should be string integers
        assert list(adata.var_names) == ["0", "1", "2", "3", "4"]
        # Accession should be kept as a var column
        assert "Accession" in adata.var.columns

    def test_no_cell_key_fallback(self, tmp_dir):
        """When col_attrs has no recognised cell key, use integer indices."""
        path = os.path.join(tmp_dir, "no_cell_key.loom")
        rng = np.random.default_rng(8)
        matrix = rng.poisson(lam=1, size=(4, 6)).astype(np.float32)

        with h5py.File(path, "w") as f:
            f.create_dataset("matrix", data=matrix)
            ra = f.create_group("row_attrs")
            ra.create_dataset("Gene", data=np.array([f"G{i}" for i in range(4)], dtype="S"))
            ca = f.create_group("col_attrs")
            ca.create_dataset("Batch", data=np.array([f"B{j}" for j in range(6)], dtype="S"))

        result = read_loom(path)
        adata = result.adata
        assert adata.shape == (6, 4)
        assert list(adata.obs_names) == ["0", "1", "2", "3", "4", "5"]
        assert "Batch" in adata.obs.columns
