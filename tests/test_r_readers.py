"""Tests for R subprocess readers (Seurat and SingleCellExperiment)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_RSCRIPT = shutil.which("Rscript")

_HAS_RSCRIPT = _RSCRIPT is not None

_HAS_SEURAT = False
_HAS_SCE = False

if _HAS_RSCRIPT:
    try:
        _ret = subprocess.run(
            [_RSCRIPT, "-e", 'library(Seurat); cat("ok")'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        _HAS_SEURAT = _ret.stdout.strip().endswith("ok")
    except Exception:
        pass

    try:
        _ret = subprocess.run(
            [_RSCRIPT, "-e", 'library(SingleCellExperiment); cat("ok")'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        _HAS_SCE = _ret.stdout.strip().endswith("ok")
    except Exception:
        pass


requires_seurat = pytest.mark.skipif(
    not (_HAS_RSCRIPT and _HAS_SEURAT),
    reason="Rscript or Seurat not available",
)

requires_sce = pytest.mark.skipif(
    not (_HAS_RSCRIPT and _HAS_SCE),
    reason="Rscript or SingleCellExperiment not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_seurat_rds(tmp_path_factory):
    """Create a minimal Seurat RDS via R subprocess."""
    if not (_HAS_RSCRIPT and _HAS_SEURAT):
        pytest.skip("Rscript or Seurat not available")

    out_dir = tmp_path_factory.mktemp("seurat_fixture")
    rds_path = str(out_dir / "tiny_seurat.rds")

    r_code = f"""
    suppressPackageStartupMessages(library(Seurat))
    set.seed(42)
    counts <- matrix(rpois(200, lambda = 3), nrow = 20, ncol = 10)
    rownames(counts) <- paste0("Gene", 1:20)
    colnames(counts) <- paste0("Cell", 1:10)
    obj <- CreateSeuratObject(counts = counts, project = "tiny")
    obj$celltype <- rep(c("TypeA", "TypeB"), each = 5)
    saveRDS(obj, "{rds_path}")
    cat("ok")
    """
    proc = subprocess.run(
        [_RSCRIPT, "-e", r_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.stdout.strip().endswith("ok"), (
        f"Failed to create Seurat fixture:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return rds_path


@pytest.fixture(scope="session")
def tiny_sce_rds(tmp_path_factory):
    """Create a minimal SingleCellExperiment RDS via R subprocess."""
    if not (_HAS_RSCRIPT and _HAS_SCE):
        pytest.skip("Rscript or SingleCellExperiment not available")

    out_dir = tmp_path_factory.mktemp("sce_fixture")
    rds_path = str(out_dir / "tiny_sce.rds")

    r_code = f"""
    suppressPackageStartupMessages(library(SingleCellExperiment))
    set.seed(42)
    counts <- matrix(rpois(200, lambda = 3), nrow = 20, ncol = 10)
    rownames(counts) <- paste0("Gene", 1:20)
    colnames(counts) <- paste0("Cell", 1:10)
    sce <- SingleCellExperiment(assays = list(counts = counts))
    sce$celltype <- rep(c("TypeA", "TypeB"), each = 5)
    saveRDS(sce, "{rds_path}")
    cat("ok")
    """
    proc = subprocess.run(
        [_RSCRIPT, "-e", r_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.stdout.strip().endswith("ok"), (
        f"Failed to create SCE fixture:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return rds_path


# ---------------------------------------------------------------------------
# Seurat tests
# ---------------------------------------------------------------------------


@requires_seurat
class TestSeuratReader:
    def test_basic_read(self, tiny_seurat_rds):
        from scripts.readers.r_bridge import read_seurat

        result = read_seurat(tiny_seurat_rds)
        adata = result.adata
        # 10 cells x 20 genes
        assert adata.shape == (10, 20), f"Expected (10, 20), got {adata.shape}"
        assert result.source_meta["source_format"] == "seurat_rds"

    def test_metadata_extracted(self, tiny_seurat_rds):
        from scripts.readers.r_bridge import read_seurat

        result = read_seurat(tiny_seurat_rds)
        adata = result.adata
        assert "celltype" in adata.obs.columns, (
            f"Expected 'celltype' in obs columns, got {list(adata.obs.columns)}"
        )
        # Check values are correct
        celltypes = adata.obs["celltype"].unique()
        assert set(celltypes) == {"TypeA", "TypeB"}

    def test_matrix_type_hint(self, tiny_seurat_rds):
        from scripts.readers.r_bridge import read_seurat

        result = read_seurat(tiny_seurat_rds)
        assert result.source_meta["matrix_type_hint"] == "counts"

    def test_assay_recorded(self, tiny_seurat_rds):
        from scripts.readers.r_bridge import read_seurat

        result = read_seurat(tiny_seurat_rds)
        assert result.source_meta["assay"] == "RNA"


# ---------------------------------------------------------------------------
# SingleCellExperiment tests
# ---------------------------------------------------------------------------


@requires_sce
class TestSCEReader:
    def test_basic_read(self, tiny_sce_rds):
        from scripts.readers.r_bridge import read_sce

        result = read_sce(tiny_sce_rds)
        adata = result.adata
        assert adata.shape == (10, 20), f"Expected (10, 20), got {adata.shape}"
        assert result.source_meta["source_format"] == "sce"

    def test_metadata_extracted(self, tiny_sce_rds):
        from scripts.readers.r_bridge import read_sce

        result = read_sce(tiny_sce_rds)
        adata = result.adata
        assert "celltype" in adata.obs.columns, (
            f"Expected 'celltype' in obs columns, got {list(adata.obs.columns)}"
        )
        celltypes = adata.obs["celltype"].unique()
        assert set(celltypes) == {"TypeA", "TypeB"}

    def test_matrix_type_hint(self, tiny_sce_rds):
        from scripts.readers.r_bridge import read_sce

        result = read_sce(tiny_sce_rds)
        assert result.source_meta["matrix_type_hint"] == "counts"

    def test_slot_hint(self, tiny_sce_rds):
        from scripts.readers.r_bridge import read_sce

        result = read_sce(tiny_sce_rds)
        assert result.source_meta["slot_hint"] == "counts"
