"""Tests for scripts/stanobj.py — main orchestrator CLI.

Each test invokes stanobj.py as a subprocess (using sys.executable),
mirroring how users will call the tool.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

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
# End-to-end tests
# ---------------------------------------------------------------------------


class TestMtxEndToEnd:
    def test_mtx_end_to_end(self, tiny_mtx_dir, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "output.h5ad")
        proc = run_stanobj(tiny_mtx_dir, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"

        # Output files exist
        assert os.path.isfile(out_h5ad)
        stem = "output"
        report_path = os.path.join(tmp_dir, f"{stem}_report.json")
        audit_path = os.path.join(tmp_dir, f"{stem}_audit.log")
        assert os.path.isfile(report_path), "report.json missing"
        assert os.path.isfile(audit_path), "audit.log missing"

        # Shape and provenance
        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape[0] == 10
        assert adata.shape[1] == 20
        assert "stanobj" in adata.uns


class TestCsvEndToEnd:
    def test_csv_end_to_end(self, tiny_csv_path, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "csv_output.h5ad")
        # The tiny CSV has low max counts (max ~9) which the classifier
        # cannot auto-resolve, so we supply matrix_type as a decision.
        proc = run_stanobj(
            tiny_csv_path, "-o", out_h5ad,
            "--decision", "matrix_type=counts",
        )
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)


class TestH5adEndToEnd:
    def test_h5ad_end_to_end(self, tiny_h5ad_path, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "h5ad_output.h5ad")
        # h5ad reader does not set matrix_type_hint, and our tiny fixture
        # has max=9 which the classifier can't auto-resolve.
        proc = run_stanobj(
            tiny_h5ad_path, "-o", out_h5ad,
            "--decision", "matrix_type=counts",
        )
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"

        adata = ad.read_h5ad(out_h5ad)
        assert "stanobj" in adata.uns


class TestH5EndToEnd:
    def test_10x_h5_end_to_end(self, tiny_10x_h5_path, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "h5_output.h5ad")
        proc = run_stanobj(tiny_10x_h5_path, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"

        adata = ad.read_h5ad(out_h5ad)
        assert adata.shape == (10, 20)


class TestDecisionExitCode:
    def test_decision_exit_code(self, tmp_dir):
        """Create an ambiguous 5x5 CSV (X0..X4, Y0..Y4) that triggers a
        matrix_orientation decision."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 10, size=(5, 5)).astype(float)
        row_labels = [f"X{i}" for i in range(5)]
        col_labels = [f"Y{i}" for i in range(5)]
        df = pd.DataFrame(data, index=row_labels, columns=col_labels)
        csv_path = os.path.join(tmp_dir, "ambiguous.csv")
        df.to_csv(csv_path, index=True)

        out_h5ad = os.path.join(tmp_dir, "ambiguous_output.h5ad")
        proc = run_stanobj(csv_path, "-o", out_h5ad)

        assert proc.returncode == 10, (
            f"Expected exit code 10 for decision_needed, got {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

        # Parse JSON from stdout
        decision = json.loads(proc.stdout)
        assert decision["status"] == "decision_needed"


class TestDecisionSupplied:
    def test_decision_supplied(self, tmp_dir):
        """Same ambiguous CSV, but supply --decision matrix_orientation=cells_x_genes."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 10, size=(5, 5)).astype(float)
        row_labels = [f"X{i}" for i in range(5)]
        col_labels = [f"Y{i}" for i in range(5)]
        df = pd.DataFrame(data, index=row_labels, columns=col_labels)
        csv_path = os.path.join(tmp_dir, "ambiguous2.csv")
        df.to_csv(csv_path, index=True)

        out_h5ad = os.path.join(tmp_dir, "decided_output.h5ad")
        proc = run_stanobj(
            csv_path, "-o", out_h5ad,
            "--decision", "matrix_orientation=cells_x_genes",
            "--decision", "matrix_type=counts",
        )

        assert proc.returncode == 0, (
            f"Expected exit 0 with decision supplied, got {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert os.path.isfile(out_h5ad)


class TestMissingOutputArg:
    def test_missing_output_arg(self, tiny_h5ad_path):
        """Running without -o should give a nonzero exit code."""
        proc = run_stanobj(tiny_h5ad_path)
        assert proc.returncode != 0


class TestFormatOverride:
    def test_format_override(self, tiny_h5ad_path, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "fmt_override.h5ad")
        proc = run_stanobj(
            tiny_h5ad_path, "-o", out_h5ad,
            "--format", "h5ad",
            "--decision", "matrix_type=counts",
        )
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"


class TestReportJsonValid:
    def test_report_json_valid(self, tiny_mtx_dir, tmp_dir):
        out_h5ad = os.path.join(tmp_dir, "report_check.h5ad")
        proc = run_stanobj(tiny_mtx_dir, "-o", out_h5ad)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"

        report_path = os.path.join(tmp_dir, "report_check_report.json")
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            report = json.load(f)

        assert report["stanobj_version"] == "1.0.0"
        assert report["n_cells_after"] == 10
        assert report["n_genes_after"] == 20
