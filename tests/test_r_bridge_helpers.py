"""Tests for r_bridge Python helpers (no R required).

These exercise the metadata/reduction-merge helpers that would otherwise
only be reached through a live R subprocess.  They guard against the
silent misalignment that occurs when the exported CSVs have rows in a
different order than ``adata.obs_names``.
"""

from __future__ import annotations

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.readers.r_bridge import _merge_metadata, _merge_reductions


def _make_adata(cell_ids):
    n = len(cell_ids)
    X = sparse.csr_matrix(
        np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    )
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


class TestMergeMetadataAlignment:
    def test_shuffled_rows_realigned_by_name(self, tmp_dir):
        """Metadata CSV rows in an order that differs from adata.obs_names
        must be realigned by cell name, not assigned positionally."""
        adata = _make_adata(["cell0", "cell1", "cell2", "cell3"])

        shuffled = pd.DataFrame(
            {"celltype": ["C", "A", "D", "B"]},
            index=["cell2", "cell0", "cell3", "cell1"],
        )
        csv_path = os.path.join(tmp_dir, "metadata.csv")
        shuffled.to_csv(csv_path)

        _merge_metadata(adata, csv_path)

        assert adata.obs.loc["cell0", "celltype"] == "A"
        assert adata.obs.loc["cell1", "celltype"] == "B"
        assert adata.obs.loc["cell2", "celltype"] == "C"
        assert adata.obs.loc["cell3", "celltype"] == "D"

    def test_matching_order_preserved(self, tmp_dir):
        """When CSV already matches adata.obs_names, values pass through unchanged."""
        adata = _make_adata(["cell0", "cell1", "cell2"])
        meta = pd.DataFrame(
            {"celltype": ["X", "Y", "Z"]},
            index=["cell0", "cell1", "cell2"],
        )
        csv_path = os.path.join(tmp_dir, "metadata.csv")
        meta.to_csv(csv_path)

        _merge_metadata(adata, csv_path)

        assert list(adata.obs["celltype"]) == ["X", "Y", "Z"]

    def test_no_name_overlap_falls_back_to_positional(self, tmp_dir):
        """If CSV index has zero overlap with obs_names but lengths match,
        fall back to positional assignment (backwards compat)."""
        adata = _make_adata(["cell0", "cell1", "cell2"])
        meta = pd.DataFrame(
            {"celltype": ["X", "Y", "Z"]},
            index=["otherA", "otherB", "otherC"],
        )
        csv_path = os.path.join(tmp_dir, "metadata.csv")
        meta.to_csv(csv_path)

        _merge_metadata(adata, csv_path)

        # Positional: cell0->X, cell1->Y, cell2->Z
        assert list(adata.obs["celltype"]) == ["X", "Y", "Z"]

    def test_missing_file_no_op(self, tmp_dir):
        adata = _make_adata(["a", "b"])
        _merge_metadata(adata, os.path.join(tmp_dir, "does_not_exist.csv"))
        assert len(adata.obs.columns) == 0


class TestMergeReductionsAlignment:
    def test_shuffled_rows_realigned_by_name(self, tmp_dir):
        """Reduction CSV whose rownames are shuffled must be reindexed
        to adata.obs_names before assignment to obsm."""
        adata = _make_adata(["cell0", "cell1", "cell2", "cell3"])

        # Shuffled index — values chosen so each row's PC1 == 10*int(suffix)
        shuffled = pd.DataFrame(
            {
                "PC_1": [20.0, 0.0, 30.0, 10.0],
                "PC_2": [21.0, 1.0, 31.0, 11.0],
            },
            index=["cell2", "cell0", "cell3", "cell1"],
        )
        csv_path = os.path.join(tmp_dir, "reduction_pca.csv")
        shuffled.to_csv(csv_path)

        names = _merge_reductions(adata, tmp_dir)

        assert "pca" in names
        assert "X_pca" in adata.obsm
        pca = np.asarray(adata.obsm["X_pca"])

        # Row 0 is cell0 -> PC_1 = 0, PC_2 = 1
        np.testing.assert_array_almost_equal(pca[0], [0.0, 1.0])
        np.testing.assert_array_almost_equal(pca[1], [10.0, 11.0])
        np.testing.assert_array_almost_equal(pca[2], [20.0, 21.0])
        np.testing.assert_array_almost_equal(pca[3], [30.0, 31.0])

    def test_matching_order_preserved(self, tmp_dir):
        adata = _make_adata(["c0", "c1"])
        aligned = pd.DataFrame(
            {"UMAP_1": [1.0, 2.0], "UMAP_2": [3.0, 4.0]},
            index=["c0", "c1"],
        )
        aligned.to_csv(os.path.join(tmp_dir, "reduction_umap.csv"))

        names = _merge_reductions(adata, tmp_dir)

        assert names == ["umap"]
        np.testing.assert_array_almost_equal(
            np.asarray(adata.obsm["X_umap"]),
            [[1.0, 3.0], [2.0, 4.0]],
        )

    def test_missing_cells_raises(self, tmp_dir):
        """If the CSV is missing some cells, the helper must not silently
        produce NaN-padded embeddings — raise loudly."""
        adata = _make_adata(["cell0", "cell1", "cell2"])
        incomplete = pd.DataFrame(
            {"PC_1": [1.0, 2.0]},
            index=["cell0", "cell1"],  # cell2 missing
        )
        incomplete.to_csv(os.path.join(tmp_dir, "reduction_pca.csv"))

        with pytest.raises((ValueError, KeyError)):
            _merge_reductions(adata, tmp_dir)

    def test_no_name_overlap_falls_back_to_positional(self, tmp_dir):
        """If CSV has zero name overlap but row count matches, fall back
        to positional (handles R scripts that didn't set rownames)."""
        adata = _make_adata(["cell0", "cell1"])
        unaligned = pd.DataFrame(
            {"PC_1": [7.0, 8.0]},
            index=["rowA", "rowB"],
        )
        unaligned.to_csv(os.path.join(tmp_dir, "reduction_pca.csv"))

        names = _merge_reductions(adata, tmp_dir)

        assert names == ["pca"]
        np.testing.assert_array_almost_equal(
            np.asarray(adata.obsm["X_pca"]).ravel(),
            [7.0, 8.0],
        )
