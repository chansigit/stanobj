"""Tests for the design.yaml sample-metadata broadcast feature.

Covers:
- ``load_design_sample_metadata`` selection rules (single sample auto,
  multi-sample with explicit ``sample_id``, multi-sample without it
  raises, unknown sample_id raises).
- ``broadcast_design_metadata`` writing obs columns + skipping existing.
- End-to-end: stanobj.py CLI with ``--design-yaml`` writes adata.h5ad
  whose obs has the broadcast columns.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse
from scipy.io import mmwrite

# Make project root importable so we can import stanobj as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.stanobj import (  # noqa: E402
    broadcast_design_metadata,
    load_design_sample_metadata,
)

STANOBJ_CLI = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "stanobj.py"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_design(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _build_adata(n_obs: int = 5, n_vars: int = 4) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(n_obs, n_vars)))
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


def _write_mtx_sample(out_dir: Path) -> Path:
    """Drop a 10x-style mtx triplet under out_dir for the CLI test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(rng.poisson(2, size=(6, 5)).astype(np.int64))
    mmwrite(str(out_dir / "matrix.mtx"), X.T)  # genes × cells per 10x
    (out_dir / "barcodes.tsv").write_text(
        "\n".join(f"AAAG{i:04d}-1" for i in range(6)) + "\n"
    )
    (out_dir / "features.tsv").write_text(
        "\n".join(f"ENSG{i:07d}\tGENE{i}\tGene Expression"
                 for i in range(5)) + "\n"
    )
    return out_dir


# ---------------------------------------------------------------------------
# load_design_sample_metadata
# ---------------------------------------------------------------------------


class TestLoadDesignSampleMetadata:
    def test_single_sample_auto_select(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "dataset_id": "GSE_X",
            "organism": "Homo sapiens",
            "samples": [
                {
                    "sample_id": "GSM1",
                    "accession": "GSM1",
                    "extra": {"tissue": "lung", "disease": "healthy"},
                }
            ],
        })
        meta, sid = load_design_sample_metadata(str(d))
        assert sid == "GSM1"
        assert meta["species"] == "Homo sapiens"
        assert meta["tissue"] == "lung"
        assert meta["disease"] == "healthy"
        assert meta["sample_id"] == "GSM1"

    def test_multi_sample_requires_sample_id(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "samples": [
                {"sample_id": "GSM1", "accession": "GSM1"},
                {"sample_id": "GSM2", "accession": "GSM2"},
            ],
        })
        with pytest.raises(ValueError, match="--sample-id"):
            load_design_sample_metadata(str(d))

    def test_multi_sample_explicit_pick(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "organism": "Mus musculus",
            "samples": [
                {"sample_id": "GSM1", "accession": "GSM1",
                 "extra": {"tissue": "brain"}},
                {"sample_id": "GSM2", "accession": "GSM2",
                 "extra": {"tissue": "liver"}},
            ],
        })
        meta, sid = load_design_sample_metadata(str(d), sample_id="GSM2")
        assert sid == "GSM2"
        assert meta["tissue"] == "liver"

    def test_unknown_sample_id_raises(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "samples": [{"sample_id": "GSM1", "accession": "GSM1"}],
        })
        with pytest.raises(ValueError, match="not in design.yaml"):
            load_design_sample_metadata(str(d), sample_id="GSM_GHOST")

    def test_drops_non_string_extras(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "samples": [{
                "sample_id": "GSM1",
                "extra": {
                    "tissue": "lung",
                    "n_cells": 1000,           # int — dropped
                    "ages": [1, 2, 3],         # list — dropped
                    "blank": "",               # empty — dropped
                    "donor_id": "D1",
                },
            }],
        })
        meta, _ = load_design_sample_metadata(str(d))
        assert meta["tissue"] == "lung"
        assert meta["donor_id"] == "D1"
        assert "n_cells" not in meta
        assert "ages" not in meta
        assert "blank" not in meta

    def test_sample_organism_overrides_top_level(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {
            "organism": "Homo sapiens",
            "samples": [{
                "sample_id": "GSM1",
                "organism": "Mus musculus",
                "extra": {},
            }],
        })
        meta, _ = load_design_sample_metadata(str(d))
        assert meta["species"] == "Mus musculus"

    def test_no_samples_raises(self, tmp_path):
        d = _write_design(tmp_path / "design.yaml", {"organism": "x"})
        with pytest.raises(ValueError, match="no 'samples'"):
            load_design_sample_metadata(str(d))


# ---------------------------------------------------------------------------
# broadcast_design_metadata
# ---------------------------------------------------------------------------


class TestBroadcastDesignMetadata:
    def test_adds_columns_to_every_cell(self):
        a = _build_adata(n_obs=5)
        prov = broadcast_design_metadata(
            a, {"tissue": "lung", "species": "Mus musculus"},
        )
        assert prov["added"] == ["species", "tissue"]
        assert prov["skipped_existing"] == []
        assert (a.obs["tissue"] == "lung").all()
        assert (a.obs["species"] == "Mus musculus").all()
        assert len(a.obs["tissue"]) == 5

    def test_skips_existing_columns(self):
        a = _build_adata(n_obs=3)
        a.obs["tissue"] = ["heart", "heart", "heart"]  # pre-existing
        prov = broadcast_design_metadata(
            a, {"tissue": "lung", "donor_id": "D1"},
        )
        assert prov["added"] == ["donor_id"]
        assert prov["skipped_existing"] == ["tissue"]
        # original tissue preserved
        assert (a.obs["tissue"] == "heart").all()
        assert (a.obs["donor_id"] == "D1").all()

    def test_empty_metadata_is_noop(self):
        a = _build_adata(n_obs=3)
        before = list(a.obs.columns)
        prov = broadcast_design_metadata(a, {})
        assert prov == {"added": [], "skipped_existing": []}
        assert list(a.obs.columns) == before


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    def test_design_yaml_broadcasts_to_obs(self, tmp_path):
        # Build mtx triplet
        sample_dir = tmp_path / "raw"
        _write_mtx_sample(sample_dir)
        # Build design.yaml with sample-level metadata
        design = _write_design(tmp_path / "design.yaml", {
            "dataset_id": "TEST_DS",
            "organism": "Mus musculus",
            "samples": [{
                "sample_id": "GSM1",
                "accession": "GSM1",
                "extra": {
                    "tissue": "esophagus",
                    "cell_type": "Epithelia, Stroma",  # multi-value preserved
                    "donor_id": "M1",
                },
            }],
        })
        out = tmp_path / "adata.h5ad"
        rc = subprocess.run(
            [sys.executable, STANOBJ_CLI,
             str(sample_dir),
             "-o", str(out),
             "--design-yaml", str(design)],
            capture_output=True, text=True,
        )
        assert rc.returncode == 0, rc.stderr
        a = ad.read_h5ad(out)
        # obs has the broadcast columns
        assert "species" in a.obs.columns
        assert "tissue" in a.obs.columns
        assert "donor_id" in a.obs.columns
        assert (a.obs["species"] == "Mus musculus").all()
        assert (a.obs["tissue"] == "esophagus").all()
        # multi-value preserved verbatim — the consumer (eca-curation
        # 04_ontology) decides whether to attempt mapping
        assert (a.obs["cell_type"] == "Epithelia, Stroma").all()

    def test_multi_sample_without_sample_id_fails(self, tmp_path):
        sample_dir = tmp_path / "raw"
        _write_mtx_sample(sample_dir)
        design = _write_design(tmp_path / "design.yaml", {
            "samples": [
                {"sample_id": "GSM1", "accession": "GSM1"},
                {"sample_id": "GSM2", "accession": "GSM2"},
            ],
        })
        out = tmp_path / "adata.h5ad"
        rc = subprocess.run(
            [sys.executable, STANOBJ_CLI,
             str(sample_dir),
             "-o", str(out),
             "--design-yaml", str(design)],
            capture_output=True, text=True,
        )
        assert rc.returncode == 1
        assert not out.exists()
        # error surfaces clearly
        assert "sample-id" in (rc.stderr + rc.stdout).lower() \
               or "samples" in (rc.stderr + rc.stdout).lower()

    def test_no_design_flag_is_backward_compat(self, tmp_path):
        """Existing callers without --design-yaml should work unchanged."""
        sample_dir = tmp_path / "raw"
        _write_mtx_sample(sample_dir)
        out = tmp_path / "adata.h5ad"
        rc = subprocess.run(
            [sys.executable, STANOBJ_CLI,
             str(sample_dir),
             "-o", str(out)],
            capture_output=True, text=True,
        )
        assert rc.returncode == 0, rc.stderr
        a = ad.read_h5ad(out)
        # No broadcast columns
        for col in ("species", "tissue", "donor_id"):
            assert col not in a.obs.columns
