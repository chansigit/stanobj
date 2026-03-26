"""End-to-end tests for compressed and archived inputs through the stanobj pipeline.

Each test creates a compressed or archived version of a tiny fixture,
invokes stanobj.py as a subprocess, and verifies the output.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

import anndata as ad
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Path to the orchestrator script
STANOBJ = os.path.join(os.path.dirname(__file__), "..", "scripts", "stanobj.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_stanobj(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run stanobj.py with the given arguments."""
    cmd = [sys.executable, STANOBJ] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Gzipped single-file inputs
# ---------------------------------------------------------------------------


class TestGzippedInputs:
    def test_csv_gz(self, tiny_csv_path, tmp_dir):
        """Gzip a CSV and run stanobj with --decision matrix_type=counts."""
        gz_path = tiny_csv_path + ".gz"
        with open(tiny_csv_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        out_h5ad = os.path.join(tmp_dir, "csv_gz_output.h5ad")
        proc = run_stanobj(
            gz_path, "-o", out_h5ad,
            "--decision", "matrix_type=counts",
        )
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert os.path.isfile(out_h5ad)

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)

    def test_h5ad_gz(self, tiny_h5ad_path, tmp_dir):
        """Gzip an h5ad file and run stanobj."""
        gz_path = tiny_h5ad_path + ".gz"
        with open(tiny_h5ad_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        out_h5ad = os.path.join(tmp_dir, "h5ad_gz_output.h5ad")
        proc = run_stanobj(
            gz_path, "-o", out_h5ad,
            "--decision", "matrix_type=counts",
        )
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert os.path.isfile(out_h5ad)


# ---------------------------------------------------------------------------
# Archive inputs (tar.gz, zip, plain tar)
# ---------------------------------------------------------------------------


class TestArchiveInputs:
    def test_tar_gz_with_mtx(self, tiny_mtx_dir, tmp_dir):
        """Create tar.gz from MTX directory files and run stanobj."""
        tar_gz_path = os.path.join(tmp_dir, "mtx_data.tar.gz")
        with tarfile.open(tar_gz_path, "w:gz") as tf:
            for fname in os.listdir(tiny_mtx_dir):
                full = os.path.join(tiny_mtx_dir, fname)
                tf.add(full, arcname=fname)

        out_h5ad = os.path.join(tmp_dir, "tar_gz_output.h5ad")
        proc = run_stanobj(tar_gz_path, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert os.path.isfile(out_h5ad)

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)

    def test_zip_with_mtx(self, tiny_mtx_dir, tmp_dir):
        """Create zip from MTX directory files and run stanobj."""
        zip_path = os.path.join(tmp_dir, "mtx_data.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for fname in os.listdir(tiny_mtx_dir):
                full = os.path.join(tiny_mtx_dir, fname)
                zf.write(full, arcname=fname)

        out_h5ad = os.path.join(tmp_dir, "zip_output.h5ad")
        proc = run_stanobj(zip_path, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert os.path.isfile(out_h5ad)

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)

    def test_plain_tar(self, tiny_mtx_dir, tmp_dir):
        """Create plain .tar from MTX directory files and run stanobj."""
        tar_path = os.path.join(tmp_dir, "mtx_data.tar")
        with tarfile.open(tar_path, "w") as tf:
            for fname in os.listdir(tiny_mtx_dir):
                full = os.path.join(tiny_mtx_dir, fname)
                tf.add(full, arcname=fname)

        out_h5ad = os.path.join(tmp_dir, "tar_output.h5ad")
        proc = run_stanobj(tar_path, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert os.path.isfile(out_h5ad)

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)
