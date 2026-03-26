"""Tests for scripts.readers.csv_reader."""

from __future__ import annotations

import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.csv_reader import read_csv
from scripts.readers.base import DecisionNeeded


# ---------------------------------------------------------------------------
# cells x genes (no transpose)
# ---------------------------------------------------------------------------


class TestCellsXGenes:
    def test_cells_x_genes(self, tiny_csv_path):
        result = read_csv(tiny_csv_path)
        adata = result.adata
        assert adata.shape == (10, 20)
        assert result.source_meta["transposed"] is False
        assert result.source_meta["source_format"] == "csv"
        assert result.source_meta["reader_used"] == "csv_reader"


# ---------------------------------------------------------------------------
# genes x cells (transposed)
# ---------------------------------------------------------------------------


class TestGenesXCellsTransposed:
    def test_genes_x_cells_transposed(self, tiny_csv_transposed_path):
        result = read_csv(tiny_csv_transposed_path)
        adata = result.adata
        assert adata.shape == (10, 20)
        assert result.source_meta["transposed"] is True


# ---------------------------------------------------------------------------
# TSV detection
# ---------------------------------------------------------------------------


class TestTsvDetection:
    def test_tsv_detection(self, tmp_dir):
        """Create a small TSV and verify tab delimiter is inferred."""
        path = os.path.join(tmp_dir, "test.tsv")
        df = pd.DataFrame(
            {"TP53": [1, 2], "BRCA1": [3, 4], "EGFR": [5, 6]},
            index=["ACGTACGTAA-1", "ACGTACGTAA-2"],
        )
        df.to_csv(path, sep="\t", index=True)
        result = read_csv(path)
        assert result.adata.shape == (2, 3)
        assert result.source_meta["source_format"] == "tsv"


# ---------------------------------------------------------------------------
# Ambiguous orientation -> DecisionNeeded
# ---------------------------------------------------------------------------


class TestDecisionNeededForAmbiguous:
    def test_decision_needed_for_ambiguous(self, tmp_dir):
        """5x5 with non-descriptive labels should raise DecisionNeeded."""
        path = os.path.join(tmp_dir, "ambiguous.csv")
        row_labels = [f"X{i}" for i in range(5)]
        col_labels = [f"Y{i}" for i in range(5)]
        df = pd.DataFrame(
            np.ones((5, 5)),
            index=row_labels,
            columns=col_labels,
        )
        df.to_csv(path, index=True)

        with pytest.raises(DecisionNeeded) as exc_info:
            read_csv(path)
        assert exc_info.value.decision_type == "matrix_orientation"
        assert "cells_x_genes" in exc_info.value.options
        assert "genes_x_cells" in exc_info.value.options


# ---------------------------------------------------------------------------
# Explicit decision resolves ambiguity
# ---------------------------------------------------------------------------


class TestExplicitDecision:
    def test_explicit_decision(self, tmp_dir):
        """Same ambiguous CSV but with explicit decision should work."""
        path = os.path.join(tmp_dir, "ambiguous2.csv")
        row_labels = [f"X{i}" for i in range(5)]
        col_labels = [f"Y{i}" for i in range(5)]
        df = pd.DataFrame(
            np.ones((5, 5)),
            index=row_labels,
            columns=col_labels,
        )
        df.to_csv(path, index=True)

        result = read_csv(path, decisions={"matrix_orientation": "cells_x_genes"})
        assert result.adata.shape == (5, 5)
        assert result.source_meta["transposed"] is False


# ---------------------------------------------------------------------------
# Gzipped CSV
# ---------------------------------------------------------------------------


class TestGzipped:
    def test_gzipped(self, tiny_csv_path):
        """Gzip the tiny CSV and verify reading still works."""
        gz_path = tiny_csv_path + ".gz"
        with open(tiny_csv_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())

        result = read_csv(gz_path)
        assert result.adata.shape == (10, 20)
        assert result.source_meta["decompressed"] is True
