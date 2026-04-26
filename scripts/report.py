"""Report and audit log generation for stanobj conversions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils import format_number, iso_now


# ---------------------------------------------------------------------------
# generate_report_json
# ---------------------------------------------------------------------------


def generate_report_json(
    source_path: str,
    source_format: str,
    source_meta: dict,
    matrix_classification: dict,
    n_cells_before: int,
    n_genes_before: int,
    n_cells_after: int,
    n_genes_after: int,
    decisions: dict,
    warnings: List[str],
) -> str:
    """Generate a JSON report summarising the conversion.

    Parameters
    ----------
    source_path
        Original input file path.
    source_format
        Detected format string (e.g. ``"h5ad"``, ``"10x_h5"``).
    source_meta
        Dict of metadata collected during reading/standardization.
    matrix_classification
        Dict with ``matrix_type``, ``confidence``, ``matrix_type_source``.
    n_cells_before, n_genes_before
        Shape before filtering/subsetting.
    n_cells_after, n_genes_after
        Shape after filtering/subsetting.
    decisions
        Dict of key decisions made during conversion.
    warnings
        List of warning strings.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    report = {
        "stanobj_version": "1.1.0",
        "source_path": source_path,
        "source_format": source_format,
        "decompressed": source_meta.get("decompressed", False),
        "reader_used": source_meta.get("reader_used", ""),
        "matrix_orientation_before": source_meta.get(
            "matrix_orientation_before", "unknown"
        ),
        "transposed": source_meta.get("transposed", False),
        "matrix_type": matrix_classification.get("matrix_type", "unknown"),
        "matrix_type_confidence": matrix_classification.get("confidence", "low"),
        "matrix_type_source": matrix_classification.get(
            "matrix_type_source", "heuristic"
        ),
        "x_contents": source_meta.get("x_contents", "unknown"),
        "layers": source_meta.get("layers", []),
        "main_assay": source_meta.get("main_assay", None),
        "raw_counts_found": source_meta.get("raw_counts_found", False),
        "feature_types_present": source_meta.get("feature_types_present", []),
        "rna_only_subset_applied": source_meta.get(
            "rna_only_subset_applied", False
        ),
        "gene_identifier_type": source_meta.get("gene_identifier_type", "unknown"),
        "duplicate_genes_resolved": source_meta.get(
            "duplicate_genes_resolved", 0
        ),
        "obs_name_strategy": source_meta.get("obs_name_strategy", None),
        "var_name_strategy": source_meta.get("var_name_strategy", None),
        "n_cells_before": n_cells_before,
        "n_genes_before": n_genes_before,
        "n_cells_after": n_cells_after,
        "n_genes_after": n_genes_after,
        "embeddings_preserved": source_meta.get("embeddings_preserved", []),
        "embeddings_dropped": source_meta.get("embeddings_dropped", []),
        "obs_columns_standardized": source_meta.get(
            "obs_columns_standardized", {}
        ),
        "decisions_made": decisions,
        "warnings": warnings,
        "errors": [],
        "conversion_timestamp": iso_now(),
    }

    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# generate_audit_log
# ---------------------------------------------------------------------------


def generate_audit_log(
    source_path: str,
    source_format: str,
    output_path: str,
    source_meta: dict,
    matrix_classification: dict,
    x_contents: str,
    n_cells: int,
    n_genes: int,
    decisions: dict,
    warnings: List[str],
) -> str:
    """Generate a human-readable audit log for the conversion.

    Parameters
    ----------
    source_path
        Original input file path.
    source_format
        Detected format string.
    output_path
        Path where the output h5ad will be written.
    source_meta
        Dict of metadata collected during reading/standardization.
    matrix_classification
        Dict with ``matrix_type``, ``confidence``, ``matrix_type_source``.
    x_contents
        What the X matrix contains after conversion.
    n_cells, n_genes
        Final shape.
    decisions
        Dict of key decisions made during conversion.
    warnings
        List of warning strings.

    Returns
    -------
    str
        Multi-line audit log string.
    """
    lines: List[str] = []

    # --- Header ---
    lines.append("=== stanobj conversion audit ===")
    lines.append(f"Source: {source_path} ({source_format})")
    lines.append(f"Output: {output_path}")
    lines.append(f"Timestamp: {iso_now()}")
    lines.append("")

    # --- Detection ---
    reader = source_meta.get("reader_used", "unknown")
    lines.append("--- Detection ---")
    lines.append(f"Format: {source_format}")
    lines.append(f"Reader: {reader}")

    assay = source_meta.get("assay_selected")
    if assay:
        lines.append(f"Assay selected: {assay}")

    if source_meta.get("decompressed", False):
        lines.append("Note: file was decompressed before reading")

    lines.append("")

    # --- Standardization ---
    orient = source_meta.get("matrix_orientation_before", "unknown")
    transposed = source_meta.get("transposed", False)
    transpose_note = "transposed" if transposed else "no transpose"

    mat_type = matrix_classification.get("matrix_type", "unknown")
    mat_conf = matrix_classification.get("confidence", "low")
    mat_source = matrix_classification.get("matrix_type_source", "heuristic")

    lines.append("--- Standardization ---")
    lines.append(f"X contents: {x_contents}")
    lines.append(f"Orientation: {orient} ({transpose_note})")
    lines.append(f"Matrix type: {mat_type} ({mat_conf}, from {mat_source})")

    dup_genes = source_meta.get("duplicate_genes_resolved", 0)
    if dup_genes:
        lines.append(f"Duplicate genes resolved: {dup_genes}")

    embeddings_preserved = source_meta.get("embeddings_preserved", [])
    embeddings_dropped = source_meta.get("embeddings_dropped", [])
    if embeddings_preserved or embeddings_dropped:
        if embeddings_preserved:
            lines.append(
                f"Embeddings preserved: {', '.join(embeddings_preserved)}"
            )
        if embeddings_dropped:
            lines.append(
                f"Embeddings dropped: {', '.join(embeddings_dropped)}"
            )

    obs_mapped = source_meta.get("obs_columns_standardized", {})
    if obs_mapped:
        mappings = [f"{k} -> {v}" for k, v in obs_mapped.items()]
        lines.append(f"Metadata columns mapped: {', '.join(mappings)}")

    lines.append("")

    # --- Decisions ---
    lines.append("--- Decisions ---")
    if decisions:
        for key, desc in decisions.items():
            lines.append(f"  {key}: {desc}")
    else:
        lines.append("  None")
    lines.append("")

    # --- Warnings ---
    lines.append("--- Warnings ---")
    if warnings:
        for w in warnings:
            lines.append(f"  {w}")
    else:
        lines.append("  None")
    lines.append("")

    # --- Validation ---
    lines.append("--- Validation ---")
    if warnings:
        n = len(warnings)
        lines.append(f"Passed with {n} warning{'s' if n != 1 else ''}.")
    else:
        lines.append("All checks passed.")
    lines.append("")
    lines.append(
        f"Shape: {format_number(n_cells)} cells x {format_number(n_genes)} genes"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------


def write_outputs(
    adata: "anndata.AnnData",
    output_path: str,
    report_json: str,
    audit_log: str,
) -> None:
    """Write the converted AnnData, report JSON, and audit log to disk.

    Three files are created alongside each other:

    - ``<output_path>``  -- the h5ad file
    - ``<stem>_report.json`` -- JSON report
    - ``<stem>_audit.log``  -- human-readable audit log

    Parameters
    ----------
    adata
        The converted AnnData object.
    output_path
        Destination path for the h5ad file.
    report_json
        JSON string produced by :func:`generate_report_json`.
    audit_log
        Audit log string produced by :func:`generate_audit_log`.
    """
    output_path = str(output_path)
    parent = os.path.dirname(output_path)
    stem = Path(output_path).stem

    # Write h5ad
    adata.write_h5ad(output_path)

    # Write report JSON alongside the h5ad
    report_path = os.path.join(parent, f"{stem}_report.json")
    with open(report_path, "w") as f:
        f.write(report_json)

    # Write audit log alongside the h5ad
    log_path = os.path.join(parent, f"{stem}_audit.log")
    with open(log_path, "w") as f:
        f.write(audit_log)
