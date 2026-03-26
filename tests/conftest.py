"""Shared pytest fixtures for stanobj tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="stanobj_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tiny matrix / ID fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_counts():
    """10x20 sparse integer count matrix (Poisson lam=2, ~60 % zeros)."""
    rng = np.random.default_rng(0)
    dense = rng.poisson(lam=2, size=(10, 20)).astype(np.float32)
    # Force roughly 60% zeros for realism
    mask = rng.random((10, 20)) < 0.6
    dense[mask] = 0
    return sparse.csr_matrix(dense)


@pytest.fixture
def tiny_cell_ids():
    """10 barcode-like cell IDs."""
    return [f"AAACCTGA-{i}" for i in range(10)]


@pytest.fixture
def tiny_gene_names():
    """20 human gene symbols with one duplicate (TP53 at index 0 and 19)."""
    genes = [
        "TP53", "BRCA1", "EGFR", "MYC", "KRAS",
        "PTEN", "RB1", "APC", "VHL", "WT1",
        "CDH1", "SMAD4", "NF1", "NF2", "RET",
        "MEN1", "PTCH1", "TSC1", "TSC2", "TP53",
    ]
    return genes


@pytest.fixture
def tiny_adata(tiny_counts, tiny_cell_ids, tiny_gene_names):
    """Minimal AnnData built from tiny fixtures (duplicates resolved)."""
    from scripts.utils import make_names_unique

    unique_genes, _ = make_names_unique(tiny_gene_names)
    adata = ad.AnnData(
        X=tiny_counts,
        obs=pd.DataFrame(index=tiny_cell_ids),
        var=pd.DataFrame(index=unique_genes),
    )
    return adata


@pytest.fixture
def tiny_h5ad_path(tiny_adata, tmp_dir):
    """Write tiny_adata to .h5ad and return the path."""
    path = os.path.join(tmp_dir, "tiny.h5ad")
    tiny_adata.write_h5ad(path)
    return path


@pytest.fixture
def tiny_mtx_dir(tiny_counts, tiny_cell_ids, tiny_gene_names, tmp_dir):
    """Write 10x-style triplet directory (genes x cells orientation).

    Directory contains:
      - matrix.mtx   (genes x cells, as per 10x convention)
      - barcodes.tsv  (one barcode per line)
      - features.tsv  (Ensembl ID <tab> symbol <tab> 'Gene Expression')
    """
    mtx_dir = os.path.join(tmp_dir, "mtx")
    os.makedirs(mtx_dir)

    # 10x stores genes-by-cells
    mat = tiny_counts.T.tocoo()
    mmwrite(os.path.join(mtx_dir, "matrix.mtx"), mat)

    with open(os.path.join(mtx_dir, "barcodes.tsv"), "w") as f:
        for bc in tiny_cell_ids:
            f.write(bc + "\n")

    with open(os.path.join(mtx_dir, "features.tsv"), "w") as f:
        for i, sym in enumerate(tiny_gene_names):
            ens_id = f"ENSG{i:011d}"
            f.write(f"{ens_id}\t{sym}\tGene Expression\n")

    return mtx_dir


@pytest.fixture
def tiny_csv_path(tiny_counts, tiny_cell_ids, tiny_gene_names, tmp_dir):
    """Write cells x genes CSV with cell IDs as first column."""
    from scripts.utils import make_names_unique

    unique_genes, _ = make_names_unique(tiny_gene_names)
    dense = tiny_counts.toarray() if sparse.issparse(tiny_counts) else np.asarray(tiny_counts)
    df = pd.DataFrame(dense, index=tiny_cell_ids, columns=unique_genes)
    path = os.path.join(tmp_dir, "tiny.csv")
    df.to_csv(path, index=True)
    return path


@pytest.fixture
def tiny_csv_transposed_path(tiny_counts, tiny_cell_ids, tiny_gene_names, tmp_dir):
    """Write genes x cells CSV (transposed)."""
    from scripts.utils import make_names_unique

    unique_genes, _ = make_names_unique(tiny_gene_names)
    dense = tiny_counts.toarray() if sparse.issparse(tiny_counts) else np.asarray(tiny_counts)
    df = pd.DataFrame(dense, index=tiny_cell_ids, columns=unique_genes)
    path = os.path.join(tmp_dir, "tiny_transposed.csv")
    df.T.to_csv(path, index=True)
    return path


@pytest.fixture
def tiny_10x_h5_path(tiny_counts, tiny_cell_ids, tiny_gene_names, tmp_dir):
    """Write 10x-style HDF5 with ``matrix/`` group containing CSC components.

    Layout mirrors Cell Ranger output::

        matrix/
            barcodes   — (n_cells,) bytes
            data       — CSC data array
            indices    — CSC row indices
            indptr     — CSC column pointers
            shape      — [n_genes, n_cells]
            features/
                id     — Ensembl IDs
                name   — gene symbols
                feature_type — "Gene Expression"
    """
    path = os.path.join(tmp_dir, "tiny_10x.h5")
    csc = sparse.csc_matrix(tiny_counts.T)  # genes x cells

    with h5py.File(path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("barcodes", data=np.array(tiny_cell_ids, dtype="S"))
        grp.create_dataset("data", data=csc.data)
        grp.create_dataset("indices", data=csc.indices)
        grp.create_dataset("indptr", data=csc.indptr)
        grp.create_dataset("shape", data=np.array(csc.shape, dtype=np.int32))

        feat = grp.create_group("features")
        ens_ids = [f"ENSG{i:011d}" for i in range(len(tiny_gene_names))]
        feat.create_dataset("id", data=np.array(ens_ids, dtype="S"))
        feat.create_dataset("name", data=np.array(tiny_gene_names, dtype="S"))
        feat.create_dataset(
            "feature_type",
            data=np.array(["Gene Expression"] * len(tiny_gene_names), dtype="S"),
        )

    return path
