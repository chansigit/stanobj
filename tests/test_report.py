"""Tests for scripts.report."""

from __future__ import annotations

import json
import os

import pytest

from scripts.report import generate_audit_log, generate_report_json, write_outputs


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def sample_source_meta():
    """Minimal source_meta dict for testing."""
    return {
        "source_path": "/data/sample.h5ad",
        "source_format": "h5ad",
        "decompressed": False,
        "reader_used": "anndata.read_h5ad",
        "matrix_orientation_before": "cells_x_genes",
        "transposed": False,
        "x_contents": "counts",
        "layers": ["counts", "log1p"],
        "main_assay": "RNA",
        "raw_counts_found": True,
        "feature_types_present": ["Gene Expression"],
        "rna_only_subset_applied": False,
        "gene_identifier_type": "symbol",
        "duplicate_genes_resolved": 2,
        "obs_name_strategy": "barcodes",
        "var_name_strategy": "gene_symbol",
        "embeddings_preserved": ["X_pca", "X_umap"],
        "embeddings_dropped": [],
        "obs_columns_standardized": {"celltype": "cell_type"},
    }


@pytest.fixture
def sample_matrix_classification():
    """Minimal matrix_classification dict."""
    return {
        "matrix_type": "counts",
        "confidence": "high",
        "matrix_type_source": "heuristic",
    }


@pytest.fixture
def sample_decisions():
    """Sample decisions dict."""
    return {
        "orient": "No transpose needed",
        "counts": "Raw counts detected in X",
    }


@pytest.fixture
def sample_warnings():
    """Sample warnings list."""
    return ["Low cell count: only 50 cells detected"]


# -----------------------------------------------------------------------
# TestGenerateReportJson
# -----------------------------------------------------------------------


class TestGenerateReportJson:
    def test_returns_valid_json(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        """Output must be valid JSON."""
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_correct_fields(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        """JSON must contain all required top-level keys."""
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)

        expected_keys = {
            "stanobj_version", "source_path", "source_format",
            "decompressed", "reader_used",
            "matrix_orientation_before", "transposed",
            "matrix_type", "matrix_type_confidence", "matrix_type_source",
            "x_contents", "layers", "main_assay",
            "raw_counts_found", "feature_types_present",
            "rna_only_subset_applied",
            "gene_identifier_type", "duplicate_genes_resolved",
            "obs_name_strategy", "var_name_strategy",
            "n_cells_before", "n_genes_before",
            "n_cells_after", "n_genes_after",
            "embeddings_preserved", "embeddings_dropped",
            "obs_columns_standardized",
            "decisions_made", "warnings", "errors",
            "conversion_timestamp",
        }
        assert expected_keys.issubset(set(parsed.keys()))

    def test_version_is_correct(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        assert parsed["stanobj_version"] == "1.0.0"

    def test_includes_warnings(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        """Warnings list must be preserved in output."""
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        assert parsed["warnings"] == sample_warnings

    def test_includes_decisions(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        """Decisions dict must be preserved in output."""
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        assert parsed["decisions_made"] == sample_decisions

    def test_errors_empty_by_default(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        assert parsed["errors"] == []

    def test_timestamp_is_iso(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        ts = parsed["conversion_timestamp"]
        assert "T" in ts

    def test_sensible_defaults_for_missing_meta(
        self, sample_matrix_classification, sample_decisions, sample_warnings,
    ):
        """Missing keys in source_meta should produce sensible defaults."""
        result = generate_report_json(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            source_meta={},
            matrix_classification=sample_matrix_classification,
            n_cells_before=100,
            n_genes_before=2000,
            n_cells_after=95,
            n_genes_after=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        parsed = json.loads(result)
        # Should still have all required keys even with empty meta
        assert "stanobj_version" in parsed
        assert parsed["decompressed"] is False
        assert parsed["transposed"] is False


# -----------------------------------------------------------------------
# TestGenerateAuditLog
# -----------------------------------------------------------------------


class TestGenerateAuditLog:
    def test_contains_header(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "=== stanobj conversion audit ===" in log

    def test_contains_source_info(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "Source:" in log
        assert "/data/sample.h5ad" in log
        assert "h5ad" in log

    def test_contains_detection_section(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "--- Detection ---" in log
        assert "Format:" in log
        assert "Reader:" in log

    def test_contains_standardization_section(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "--- Standardization ---" in log
        assert "X contents:" in log
        assert "Matrix type:" in log

    def test_contains_decisions_section(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "--- Decisions ---" in log

    def test_contains_warnings_section(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "--- Warnings ---" in log
        assert "Low cell count" in log

    def test_contains_validation_section(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "--- Validation ---" in log
        assert "All checks passed." in log

    def test_contains_shape(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions, sample_warnings,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=sample_warnings,
        )
        assert "Shape:" in log
        assert "95" in log
        assert "1,800" in log

    def test_no_warnings_shows_none(
        self, sample_source_meta, sample_matrix_classification,
        sample_decisions,
    ):
        log = generate_audit_log(
            source_path="/data/sample.h5ad",
            source_format="h5ad",
            output_path="/out/sample_stanobj.h5ad",
            source_meta=sample_source_meta,
            matrix_classification=sample_matrix_classification,
            x_contents="counts",
            n_cells=95,
            n_genes=1800,
            decisions=sample_decisions,
            warnings=[],
        )
        # After the --- Warnings --- header, should say "None"
        idx = log.index("--- Warnings ---")
        warnings_section = log[idx:idx + 200]
        assert "None" in warnings_section


# -----------------------------------------------------------------------
# TestWriteOutputs
# -----------------------------------------------------------------------


class TestWriteOutputs:
    def test_writes_all_three_files(self, tiny_adata, tmp_dir):
        """write_outputs should create .h5ad, _report.json, and _audit.log."""
        output_path = os.path.join(tmp_dir, "result.h5ad")
        report_json = '{"stanobj_version": "1.0.0"}'
        audit_log = "=== stanobj conversion audit ===\nDone."

        write_outputs(tiny_adata, output_path, report_json, audit_log)

        assert os.path.isfile(output_path)
        assert os.path.isfile(os.path.join(tmp_dir, "result_report.json"))
        assert os.path.isfile(os.path.join(tmp_dir, "result_audit.log"))

    def test_report_json_content(self, tiny_adata, tmp_dir):
        """The written report JSON should match what was passed in."""
        output_path = os.path.join(tmp_dir, "out.h5ad")
        report_json = '{"stanobj_version": "1.0.0", "key": "value"}'
        audit_log = "audit text"

        write_outputs(tiny_adata, output_path, report_json, audit_log)

        report_path = os.path.join(tmp_dir, "out_report.json")
        with open(report_path) as f:
            content = f.read()
        assert content == report_json

    def test_audit_log_content(self, tiny_adata, tmp_dir):
        """The written audit log should match what was passed in."""
        output_path = os.path.join(tmp_dir, "out.h5ad")
        report_json = "{}"
        audit_log = "=== stanobj conversion audit ===\nAll good."

        write_outputs(tiny_adata, output_path, report_json, audit_log)

        log_path = os.path.join(tmp_dir, "out_audit.log")
        with open(log_path) as f:
            content = f.read()
        assert content == audit_log

    def test_h5ad_is_readable(self, tiny_adata, tmp_dir):
        """The written h5ad should be readable by anndata."""
        import anndata as ad

        output_path = os.path.join(tmp_dir, "out.h5ad")
        write_outputs(tiny_adata, output_path, "{}", "log")

        loaded = ad.read_h5ad(output_path)
        assert loaded.shape == tiny_adata.shape
