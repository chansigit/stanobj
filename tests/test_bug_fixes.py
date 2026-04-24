"""Regression tests for the medium/low-severity bug fixes."""

from __future__ import annotations

import gzip
import os
import subprocess
import sys
import tarfile

import anndata as ad
import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STANOBJ = os.path.join(os.path.dirname(__file__), "..", "scripts", "stanobj.py")


# ---------------------------------------------------------------------------
# compute_evidence must not leak NaN into the evidence dict
# ---------------------------------------------------------------------------


class TestComputeEvidenceNaN:
    def test_nan_values_do_not_leak_into_evidence(self):
        from scripts.detection import compute_evidence

        X = np.array(
            [[1.0, 2.0, np.nan], [np.nan, 3.0, 4.0], [5.0, 6.0, 7.0]],
            dtype=float,
        )
        ev = compute_evidence(X)
        for key in ("max_value", "min_value", "mean_value"):
            assert np.isfinite(ev[key]), f"evidence[{key!r}]={ev[key]} is not finite"

    def test_all_nan_column_handled(self):
        from scripts.detection import compute_evidence

        X = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=float)
        ev = compute_evidence(X)
        assert np.isfinite(ev["max_value"])
        assert np.isfinite(ev["min_value"])


# ---------------------------------------------------------------------------
# dataset_name should strip compound extensions
# ---------------------------------------------------------------------------


class TestDatasetNameCompoundExt:
    def test_tar_gz_input_yields_stem_not_stem_tar(self, tmp_dir, tiny_mtx_dir):
        archive = os.path.join(tmp_dir, "my_data.tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            for f in os.listdir(tiny_mtx_dir):
                tf.add(os.path.join(tiny_mtx_dir, f), arcname=f)

        out = os.path.join(tmp_dir, "out.h5ad")
        proc = subprocess.run(
            [sys.executable, STANOBJ, archive, "-o", out],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"stdout:{proc.stdout}\nstderr:{proc.stderr}"
        )

        adata = ad.read_h5ad(out)
        # dataset column should be "my_data", not "my_data.tar"
        assert adata.obs["dataset"].iloc[0] == "my_data", (
            f"Expected 'my_data', got {adata.obs['dataset'].iloc[0]!r}"
        )


# ---------------------------------------------------------------------------
# detection._inspect_hdf5 must not rely on sys.path manipulated by stanobj.py
# ---------------------------------------------------------------------------


class TestDetectionImportIndependence:
    def test_detect_format_works_on_compressed_h5_standalone(self, tmp_dir):
        h5_path = os.path.join(tmp_dir, "t.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("X", data=np.zeros((3, 3), dtype=np.float32))

        gz_path = h5_path + ".gz"
        with open(h5_path, "rb") as fin, gzip.open(gz_path, "wb") as fout:
            fout.write(fin.read())

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        # Imports scripts.detection WITHOUT also adding scripts/ to sys.path
        # (which is what stanobj.py does). This exercises the in-function
        # "from utils import decompress_to_temp" bug.
        code = (
            f"import sys;"
            f"sys.path.insert(0, {project_root!r});"
            f"from scripts.detection import detect_format;"
            f"print(detect_format({gz_path!r}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"stdout:{proc.stdout}\nstderr:{proc.stderr}"
        )


# ---------------------------------------------------------------------------
# Audit log differentiates "passed" from "passed with warnings"
# ---------------------------------------------------------------------------


class TestAuditLogWarnings:
    def _make_log(self, warnings):
        from scripts.report import generate_audit_log

        return generate_audit_log(
            source_path="x.csv",
            source_format="csv",
            output_path="out.h5ad",
            source_meta={},
            matrix_classification={
                "matrix_type": "counts",
                "confidence": "high",
                "matrix_type_source": "heuristic",
            },
            x_contents="counts",
            n_cells=100,
            n_genes=200,
            decisions={},
            warnings=warnings,
        )

    @staticmethod
    def _validation_summary(log: str) -> str:
        lines = log.splitlines()
        idx = lines.index("--- Validation ---")
        return lines[idx + 1]

    def test_validation_summary_differs_with_warnings(self):
        """The single-line validation summary must differ between a
        fully-clean run and one that produced warnings."""
        clean = self._validation_summary(self._make_log(warnings=[]))
        noisy = self._validation_summary(
            self._make_log(warnings=["gene names look weird"])
        )
        assert clean != noisy, (
            f"Validation line is the same for clean vs warned runs: {clean!r}"
        )

    def test_validation_summary_clean_run(self):
        line = self._validation_summary(self._make_log(warnings=[]))
        assert "passed" in line.lower()
        assert "warning" not in line.lower()

    def test_validation_summary_with_warnings_mentions_count(self):
        line = self._validation_summary(
            self._make_log(warnings=["a", "b", "c"])
        )
        # The summary should indicate how many warnings surfaced.
        assert "3" in line
