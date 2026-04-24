"""Tests for scripts.readers.mtx_reader."""

from __future__ import annotations

import gzip
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.mtx_reader import read_mtx


# ---------------------------------------------------------------------------
# Basic triplet directory
# ---------------------------------------------------------------------------


class TestBasicTripletDir:
    def test_basic_triplet_dir(self, tiny_mtx_dir):
        result = read_mtx(tiny_mtx_dir)
        adata = result.adata
        # matrix.mtx is 20 genes x 10 cells; after transpose -> (10, 20)
        assert adata.shape == (10, 20)
        assert result.source_meta["source_format"] == "mtx"
        assert result.source_meta["transposed"] is True

    def test_cells_x_genes_orientation(self, tiny_mtx_dir):
        result = read_mtx(tiny_mtx_dir)
        adata = result.adata
        # obs index should contain barcode-like cell IDs
        assert any("AAACCTGA" in name for name in adata.obs_names)

    def test_feature_types_preserved(self, tiny_mtx_dir):
        result = read_mtx(tiny_mtx_dir)
        adata = result.adata
        assert "feature_type" in adata.var.columns
        assert adata.var["feature_type"].iloc[0] == "Gene Expression"

    def test_gene_id_and_symbol(self, tiny_mtx_dir):
        result = read_mtx(tiny_mtx_dir)
        adata = result.adata
        assert "gene_id" in adata.var.columns
        assert "gene_symbol" in adata.var.columns


# ---------------------------------------------------------------------------
# Two-column features file
# ---------------------------------------------------------------------------


class TestTwoColumnFeatures:
    def test_two_column_features(self, tmp_dir):
        """Create a 5-gene x 3-cell MTX with 2-column features.tsv."""
        mtx_dir = os.path.join(tmp_dir, "two_col_mtx")
        os.makedirs(mtx_dir)

        rng = np.random.default_rng(42)
        mat = sparse.random(5, 3, density=0.5, format="coo", random_state=42)
        mat.data = rng.integers(1, 10, size=mat.data.shape).astype(np.float64)
        mmwrite(os.path.join(mtx_dir, "matrix.mtx"), mat)

        with open(os.path.join(mtx_dir, "barcodes.tsv"), "w") as f:
            for i in range(3):
                f.write(f"CELL-{i}\n")

        # Only 2 columns: gene_id and gene_symbol (no feature_type)
        with open(os.path.join(mtx_dir, "features.tsv"), "w") as f:
            for i in range(5):
                f.write(f"ENSG{i:011d}\tGENE{i}\n")

        result = read_mtx(mtx_dir)
        adata = result.adata
        assert adata.shape == (3, 5)
        assert "feature_type" in adata.var.columns
        assert adata.var["feature_type"].iloc[0] == "Gene Expression"


# ---------------------------------------------------------------------------
# GEO-style prefixed triplet (multi-library deposits)
# ---------------------------------------------------------------------------


class TestGEOPrefixedTriplet:
    """GEO deposits often prefix each file with <GSM>_<library>_ ; e.g.
    GSM123456_exp1_matrix.mtx.gz. stanobj must read such directories as-is,
    without relying on external symlink workarounds.
    """

    def _write_triplet(self, mtx_dir, prefix, counts, cell_ids, gene_names):
        os.makedirs(mtx_dir, exist_ok=True)
        mat = counts.T.tocoo()
        mmwrite(os.path.join(mtx_dir, f"{prefix}matrix.mtx"), mat)
        with open(os.path.join(mtx_dir, f"{prefix}barcodes.tsv"), "w") as f:
            for bc in cell_ids:
                f.write(bc + "\n")
        with open(os.path.join(mtx_dir, f"{prefix}features.tsv"), "w") as f:
            for i, sym in enumerate(gene_names):
                f.write(f"ENSG{i:011d}\t{sym}\tGene Expression\n")

    def test_geo_prefixed_triplet(self, tiny_counts, tiny_cell_ids,
                                  tiny_gene_names, tmp_dir):
        mtx_dir = os.path.join(tmp_dir, "GSM123456")
        self._write_triplet(
            mtx_dir, "GSM123456_exp1_spleen_l1_",
            tiny_counts, tiny_cell_ids, tiny_gene_names,
        )
        result = read_mtx(mtx_dir)
        assert result.adata.shape == (10, 20)
        assert result.source_meta["source_format"] == "mtx"
        # Non-GEO-prefix bare names should still take precedence when
        # both coexist (guardrail against glob fallback stealing
        # priority); that is the subject of the next test.

    def test_bare_names_precedence_over_prefixed(self, tiny_counts,
                                                 tiny_cell_ids, tiny_gene_names,
                                                 tmp_dir):
        """If both bare CellRanger names and GEO-prefixed names coexist,
        bare names win — they are the canonical form."""
        mtx_dir = os.path.join(tmp_dir, "mixed")
        os.makedirs(mtx_dir)
        # Bare names — correct content (our tiny fixture).
        self._write_triplet(
            mtx_dir, "",  # no prefix
            tiny_counts, tiny_cell_ids, tiny_gene_names,
        )
        # Distractors with a different shape under a GEO prefix — if the
        # reader ever picks these, read_mtx will raise on shape mismatch.
        distractor = sparse.random(3, 2, density=1.0, format="coo",
                                   random_state=0)
        mmwrite(os.path.join(mtx_dir, "GSM_other_matrix.mtx"), distractor)
        result = read_mtx(mtx_dir)
        assert result.adata.shape == (10, 20)  # bare names, not distractor

    def test_geo_prefixed_ambiguous_raises(self, tiny_counts, tiny_cell_ids,
                                           tiny_gene_names, tmp_dir):
        """Two competing prefixed libraries in one directory → ambiguous."""
        mtx_dir = os.path.join(tmp_dir, "ambiguous")
        os.makedirs(mtx_dir)
        mat = tiny_counts.T.tocoo()
        mmwrite(os.path.join(mtx_dir, "libA_matrix.mtx"), mat)
        mmwrite(os.path.join(mtx_dir, "libB_matrix.mtx"), mat)
        with open(os.path.join(mtx_dir, "libA_barcodes.tsv"), "w") as f:
            for bc in tiny_cell_ids:
                f.write(bc + "\n")
        with open(os.path.join(mtx_dir, "libA_features.tsv"), "w") as f:
            for i, sym in enumerate(tiny_gene_names):
                f.write(f"ENSG{i:011d}\t{sym}\tGene Expression\n")
        with pytest.raises(ValueError, match="Ambiguous"):
            read_mtx(mtx_dir)


# ---------------------------------------------------------------------------
# genes.tsv fallback
# ---------------------------------------------------------------------------


class TestGenesTsvFallback:
    def test_genes_tsv_fallback(self, tmp_dir):
        """Use genes.tsv instead of features.tsv."""
        mtx_dir = os.path.join(tmp_dir, "genes_fallback")
        os.makedirs(mtx_dir)

        mat = sparse.random(4, 3, density=0.5, format="coo", random_state=7)
        mmwrite(os.path.join(mtx_dir, "matrix.mtx"), mat)

        with open(os.path.join(mtx_dir, "barcodes.tsv"), "w") as f:
            for i in range(3):
                f.write(f"BC-{i}\n")

        # Use genes.tsv (not features.tsv)
        with open(os.path.join(mtx_dir, "genes.tsv"), "w") as f:
            for i in range(4):
                f.write(f"ENSG{i:011d}\tGENE{i}\n")

        result = read_mtx(mtx_dir)
        assert result.adata.shape == (3, 4)
        assert result.source_meta["source_format"] == "mtx"


# ---------------------------------------------------------------------------
# Gzipped files
# ---------------------------------------------------------------------------


class TestGzippedFiles:
    def test_gzipped_files(self, tiny_mtx_dir):
        """Gzip all files in tiny_mtx_dir, then verify reading works."""
        for fname in os.listdir(tiny_mtx_dir):
            fpath = os.path.join(tiny_mtx_dir, fname)
            gz_path = fpath + ".gz"
            with open(fpath, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(fpath)  # remove uncompressed version

        result = read_mtx(tiny_mtx_dir)
        assert result.adata.shape == (10, 20)
        assert result.source_meta["decompressed"] is True


# ---------------------------------------------------------------------------
# Single .mtx file path
# ---------------------------------------------------------------------------


class TestSingleMtxFile:
    def test_single_mtx_file(self, tiny_mtx_dir):
        """Pass matrix.mtx path directly instead of directory."""
        mtx_file = os.path.join(tiny_mtx_dir, "matrix.mtx")
        result = read_mtx(mtx_file)
        assert result.adata.shape == (10, 20)
        assert result.source_meta["source_format"] == "mtx"
