"""Tests for scripts.standardize."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as ad

from scripts.standardize import (
    OBS_COLUMN_MAP,
    standardize_obs,
    standardize_var,
    standardize_obsm,
    assign_layers,
    add_provenance,
    ensure_sparse,
)


# -----------------------------------------------------------------------
# standardize_obs
# -----------------------------------------------------------------------


class TestStandardizeObs:
    def test_adds_cell_id(self):
        obs = pd.DataFrame({"nGene": [100, 200]}, index=["AACG", "TTCG"])
        result = standardize_obs(obs, "ds1")
        assert "cell_id" in result.columns
        assert list(result["cell_id"]) == ["AACG", "TTCG"]

    def test_maps_celltype_to_cell_type(self):
        obs = pd.DataFrame({"celltype": ["T", "B"]}, index=["c1", "c2"])
        result = standardize_obs(obs, "ds1")
        assert "cell_type" in result.columns
        assert list(result["cell_type"]) == ["T", "B"]
        # original column preserved
        assert "celltype" in result.columns

    def test_maps_orig_ident_to_sample(self):
        obs = pd.DataFrame({"orig.ident": ["s1", "s2"]}, index=["c1", "c2"])
        result = standardize_obs(obs, "ds1")
        assert "sample" in result.columns
        assert list(result["sample"]) == ["s1", "s2"]

    def test_maps_patient_to_donor(self):
        obs = pd.DataFrame({"patient": ["P1", "P2"]}, index=["c1", "c2"])
        result = standardize_obs(obs, "ds1")
        assert "donor" in result.columns
        assert list(result["donor"]) == ["P1", "P2"]

    def test_maps_disease_to_condition(self):
        obs = pd.DataFrame({"disease": ["AML", "healthy"]}, index=["c1", "c2"])
        result = standardize_obs(obs, "ds1")
        assert "condition" in result.columns
        assert list(result["condition"]) == ["AML", "healthy"]

    def test_no_overwrite_existing_cell_type(self):
        obs = pd.DataFrame(
            {"cell_type": ["Treg", "Bcell"], "celltype": ["T", "B"]},
            index=["c1", "c2"],
        )
        result = standardize_obs(obs, "ds1")
        # canonical cell_type already present — should NOT be overwritten
        assert list(result["cell_type"]) == ["Treg", "Bcell"]

    def test_uniquify_obs_names(self):
        obs = pd.DataFrame({"x": [1, 2, 3]}, index=["dup", "dup", "unique"])
        result = standardize_obs(obs, "ds1", make_unique=True)
        assert result.index.is_unique

    def test_adds_dataset_column(self):
        obs = pd.DataFrame({"x": [1]}, index=["c1"])
        result = standardize_obs(obs, "my_dataset")
        assert "dataset" in result.columns
        assert result["dataset"].iloc[0] == "my_dataset"


# -----------------------------------------------------------------------
# standardize_var
# -----------------------------------------------------------------------


class TestStandardizeVar:
    def test_preserves_gene_symbol(self):
        var = pd.DataFrame({"gene_symbol": ["TP53", "BRCA1"]}, index=["g1", "g2"])
        result = standardize_var(var)
        assert list(result["gene_symbol"]) == ["TP53", "BRCA1"]

    def test_adds_gene_symbol_from_index(self):
        var = pd.DataFrame(index=["TP53", "BRCA1"])
        result = standardize_var(var)
        assert "gene_symbol" in result.columns
        assert list(result["gene_symbol"]) == ["TP53", "BRCA1"]

    def test_handles_duplicates(self):
        var = pd.DataFrame(index=["TP53", "BRCA1", "TP53"])
        result = standardize_var(var)
        assert result.index.is_unique
        assert "original_gene_name" in result.columns
        assert list(result["original_gene_name"]) == ["TP53", "BRCA1", "TP53"]

    def test_preserves_feature_type(self):
        var = pd.DataFrame(
            {"feature_type": ["Gene Expression", "Gene Expression"]},
            index=["TP53", "BRCA1"],
        )
        result = standardize_var(var)
        assert "feature_type" in result.columns
        assert list(result["feature_type"]) == ["Gene Expression", "Gene Expression"]

    def test_preserves_existing_gene_id(self):
        var = pd.DataFrame(
            {"gene_id": ["ENSG001", "ENSG002"]},
            index=["TP53", "BRCA1"],
        )
        result = standardize_var(var)
        assert "gene_id" in result.columns
        assert list(result["gene_id"]) == ["ENSG001", "ENSG002"]

    def test_return_rename_map(self):
        var = pd.DataFrame(index=["TP53", "BRCA1", "TP53"])
        result, rmap = standardize_var(var, return_rename_map=True)
        assert isinstance(rmap, dict)
        assert result.index.is_unique

    def test_no_rename_map_by_default(self):
        var = pd.DataFrame(index=["TP53", "BRCA1"])
        result = standardize_var(var)
        assert isinstance(result, pd.DataFrame)


# -----------------------------------------------------------------------
# standardize_obsm
# -----------------------------------------------------------------------


class TestStandardizeObsm:
    def test_renames_common_keys(self):
        obsm = {
            "pca": np.zeros((5, 2)),
            "umap": np.zeros((5, 2)),
            "tsne": np.zeros((5, 2)),
        }
        result = standardize_obsm(obsm)
        assert "X_pca" in result
        assert "X_umap" in result
        assert "X_tsne" in result
        assert "pca" not in result
        assert "umap" not in result
        assert "tsne" not in result

    def test_preserves_already_standard(self):
        obsm = {
            "X_pca": np.zeros((5, 2)),
            "X_umap": np.zeros((5, 2)),
        }
        result = standardize_obsm(obsm)
        assert "X_pca" in result
        assert "X_umap" in result

    def test_renames_uppercase(self):
        obsm = {
            "PCA": np.zeros((5, 2)),
            "UMAP": np.zeros((5, 2)),
            "tSNE": np.zeros((5, 2)),
        }
        result = standardize_obsm(obsm)
        assert "X_pca" in result
        assert "X_umap" in result
        assert "X_tsne" in result

    def test_passthrough_unknown_keys(self):
        obsm = {"custom_embedding": np.zeros((5, 2))}
        result = standardize_obsm(obsm)
        assert "custom_embedding" in result


# -----------------------------------------------------------------------
# assign_layers
# -----------------------------------------------------------------------


class TestAssignLayers:
    def test_counts_only(self):
        X = sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        adata = ad.AnnData(X=X)
        result = assign_layers(adata, "counts", {})
        assert "counts" in result.layers
        np.testing.assert_array_equal(result.layers["counts"].toarray(), X.toarray())

    def test_counts_and_log1p(self):
        X = sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        log_data = sparse.csr_matrix(np.log1p(X.toarray()))
        adata = ad.AnnData(X=X)
        source_layers = {"log1p": log_data}
        result = assign_layers(adata, "counts", source_layers)
        # counts should be in layers
        assert "counts" in result.layers
        # X should now be the log1p data
        np.testing.assert_array_almost_equal(
            result.X.toarray() if sparse.issparse(result.X) else result.X,
            log_data.toarray(),
        )

    def test_processed_only(self):
        X = sparse.csr_matrix(np.log1p(np.array([[1, 2], [3, 4]], dtype=np.float32)))
        raw_counts = sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        adata = ad.AnnData(X=X)
        source_layers = {"raw_counts": raw_counts}
        result = assign_layers(adata, "log-normalized", source_layers)
        # counts layer should come from raw_counts
        assert "counts" in result.layers
        np.testing.assert_array_equal(
            result.layers["counts"].toarray(), raw_counts.toarray()
        )


# -----------------------------------------------------------------------
# ensure_sparse
# -----------------------------------------------------------------------


class TestEnsureSparse:
    def test_dense_to_sparse(self):
        X = np.random.rand(15000, 50).astype(np.float32)
        adata = ad.AnnData(X=X)
        converted = ensure_sparse(adata, threshold=10000)
        assert converted is True
        assert sparse.issparse(adata.X)

    def test_small_dense_stays(self):
        X = np.random.rand(100, 50).astype(np.float32)
        adata = ad.AnnData(X=X)
        converted = ensure_sparse(adata, threshold=10000)
        assert converted is False
        assert not sparse.issparse(adata.X)

    def test_sparse_stays_sparse(self):
        X = sparse.csr_matrix(np.random.rand(15000, 50).astype(np.float32))
        adata = ad.AnnData(X=X)
        converted = ensure_sparse(adata, threshold=10000)
        assert converted is False
        assert sparse.issparse(adata.X)
