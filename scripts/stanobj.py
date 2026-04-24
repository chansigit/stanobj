#!/usr/bin/env python3
"""stanobj — convert heterogeneous single-cell data to canonical h5ad.

Usage:
    python stanobj.py <input> -o <output.h5ad> [--format <fmt>] [--decision key=value ...]

Exit codes:
    0  = success
    1  = fatal error
    10 = decision needed (JSON on stdout)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings as _warnings

# ---------------------------------------------------------------------------
# Make sibling imports work when invoked as a script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from scipy import sparse

from scripts.utils import (
    cleanup_temp,
    decompress_to_temp,
    extract_archive,
    is_archive,
    sample_matrix,
    strip_compression_ext,
)
from scripts.detection import (
    classify_with_override,
    compute_evidence,
    detect_format,
)
from scripts.validation import validate_adata
from scripts.standardize import (
    add_provenance,
    assign_layers,
    ensure_sparse,
    standardize_obs,
    standardize_obsm,
    standardize_var,
)
from scripts.report import (
    generate_audit_log,
    generate_report_json,
    write_outputs,
)
from scripts.readers.base import DecisionNeeded, ReaderResult


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stanobj",
        description="Convert single-cell data to canonical h5ad format.",
    )
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument(
        "-o", "--output", required=True, help="Output .h5ad path"
    )
    parser.add_argument(
        "--format",
        dest="format_override",
        default=None,
        help="Override auto-detected format",
    )
    parser.add_argument(
        "--decision",
        action="append",
        default=[],
        help="Repeatable key=value decision pairs",
    )
    return parser


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def parse_decisions(decision_args: list[str]) -> dict:
    """Parse ["key=value", ...] into a dict."""
    decisions: dict[str, str] = {}
    for item in decision_args:
        if "=" not in item:
            raise ValueError(f"Invalid decision format (expected key=value): {item!r}")
        key, value = item.split("=", 1)
        decisions[key.strip()] = value.strip()
    return decisions


def resolve_input(path: str) -> tuple[str, bool, list[str]]:
    """Resolve the input path, handling archives and compression.

    Returns
    -------
    resolved_path : str
        Path to the actual data (file or directory).
    was_decompressed : bool
        True if decompression was performed.
    temp_paths : list[str]
        Temporary paths to clean up afterwards.
    """
    temp_paths: list[str] = []

    # 1. Handle archives
    if is_archive(path):
        extracted_dir = extract_archive(path)
        temp_paths.append(extracted_dir)
        # If single subdirectory, descend into it — but only if it's a
        # real directory, not a symlink (defense-in-depth against archive
        # entries that escape the extraction root).
        entries = os.listdir(extracted_dir)
        if len(entries) == 1:
            sub = os.path.join(extracted_dir, entries[0])
            if os.path.isdir(sub) and not os.path.islink(sub):
                return sub, False, temp_paths
        return extracted_dir, False, temp_paths

    # 2. Handle compressed files that need seekable access
    if os.path.isfile(path):
        stripped, was_compressed = strip_compression_ext(path)
        if was_compressed:
            ext = os.path.splitext(stripped)[1].lower()
            # Only decompress formats that need seekable access
            if ext in (".h5", ".hdf5", ".h5ad", ".rds"):
                tmp = decompress_to_temp(path)
                temp_paths.append(tmp)
                return tmp, True, temp_paths
            # .csv.gz, .tsv.gz, .mtx.gz — readers handle natively
            return path, False, temp_paths

    return path, False, temp_paths


def load_source(path: str, fmt: str, decisions: dict) -> ReaderResult:
    """Dispatch to the appropriate reader based on format string."""
    from scripts.readers.mtx_reader import read_mtx
    from scripts.readers.csv_reader import read_csv
    from scripts.readers.h5_reader import read_10x_h5, read_generic_h5
    from scripts.readers.h5ad_reader import read_h5ad
    from scripts.readers.loom_reader import read_loom
    from scripts.readers.r_bridge import read_seurat, read_sce

    readers = {
        "mtx": lambda: read_mtx(path, decisions=decisions),
        "csv": lambda: read_csv(path, delimiter=",", decisions=decisions),
        "tsv": lambda: read_csv(path, delimiter="\t", decisions=decisions),
        "10x_h5": lambda: read_10x_h5(path, decisions=decisions),
        "generic_h5": lambda: read_generic_h5(path, decisions=decisions),
        "h5ad": lambda: read_h5ad(path, decisions=decisions),
        "loom": lambda: read_loom(path, decisions=decisions),
        "seurat_rds": lambda: read_seurat(path, decisions=decisions),
        "sce": lambda: read_sce(path, decisions=decisions),
    }

    if fmt not in readers:
        raise ValueError(f"Unsupported format: {fmt!r}")

    return readers[fmt]()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Main pipeline: read, classify, standardize, validate, write.

    Returns the exit code.
    """
    temp_paths: list[str] = []
    all_warnings: list[str] = []
    all_decisions: dict[str, str] = {}

    try:
        # -- Parse decisions ---------------------------------------------------
        decisions = parse_decisions(args.decision)
        all_decisions.update(decisions)

        # -- 1. Resolve input --------------------------------------------------
        resolved_path, was_decompressed, tmps = resolve_input(args.input)
        temp_paths.extend(tmps)

        # -- 2. Detect format --------------------------------------------------
        if args.format_override:
            fmt = args.format_override
        else:
            fmt = detect_format(resolved_path)

        # -- 3. Load source ----------------------------------------------------
        result: ReaderResult = load_source(resolved_path, fmt, decisions)
        adata = result.adata
        source_meta = result.source_meta

        source_meta["source_path"] = args.input
        source_meta["source_format"] = fmt
        source_meta["decompressed"] = was_decompressed

        # Gather reader warnings
        if "warnings" in source_meta:
            all_warnings.extend(source_meta["warnings"])

        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars

        # -- 4. Classify matrix type -------------------------------------------
        sampled = sample_matrix(adata.X)
        evidence = compute_evidence(sampled)
        hint = source_meta.get("matrix_type_hint")
        classification = classify_with_override(evidence, source_hint=hint)
        matrix_type = classification["matrix_type"]
        confidence = classification["confidence"]

        if classification.get("warnings"):
            all_warnings.extend(classification["warnings"])

        # -- 5. If unknown + low confidence, request decision ------------------
        if (
            matrix_type == "unknown"
            and confidence == "low"
            and "matrix_type" not in decisions
        ):
            raise DecisionNeeded(
                decision_type="matrix_type",
                context=(
                    f"Could not determine matrix type for {args.input}. "
                    f"Evidence: max={evidence['max_value']}, "
                    f"is_integer={evidence['is_integer']}, "
                    f"has_negatives={evidence['has_negatives']}."
                ),
                options=["counts", "log1p", "normalized", "scaled", "unknown"],
                reason="Heuristic classifier returned 'unknown' with low confidence.",
            )

        # -- 6. Override from decision -----------------------------------------
        if "matrix_type" in decisions:
            matrix_type = decisions["matrix_type"]
            classification["matrix_type"] = matrix_type
            classification["matrix_type_source"] = "user_decision"
            all_decisions["matrix_type"] = matrix_type

        # -- 7. Modality filter ------------------------------------------------
        if "feature_type" in adata.var.columns:
            unique_types = adata.var["feature_type"].unique().tolist()
            if len(unique_types) > 1:
                feature_types_present = sorted(unique_types)
                source_meta["feature_types_present"] = feature_types_present

                modality = decisions.get("modality_filter")
                if modality is None:
                    # Default to rna_only if Gene Expression is present
                    if "Gene Expression" in unique_types:
                        modality = "rna_only"
                        all_warnings.append(
                            f"Multiple feature types found ({feature_types_present}); "
                            f"defaulting to RNA-only subset."
                        )
                    else:
                        raise DecisionNeeded(
                            decision_type="modality_filter",
                            context=(
                                f"Multiple feature types detected: {feature_types_present}. "
                                f"No 'Gene Expression' type found."
                            ),
                            options=["rna_only", "keep_all"],
                            reason="Multiple feature types present.",
                        )

                if modality == "rna_only":
                    mask = adata.var["feature_type"] == "Gene Expression"
                    adata = adata[:, mask].copy()
                    source_meta["rna_only_subset_applied"] = True
                    source_meta["modality_filter"] = "rna_only"
                elif modality == "keep_all":
                    all_warnings.append(
                        "Keeping all feature types (multi-modal). "
                        "Downstream analyses may need modality-specific handling."
                    )
                    source_meta["modality_filter"] = "keep_all"

        # -- 8. Standardize obs ------------------------------------------------
        dataset_name = os.path.splitext(os.path.basename(args.input))[0]
        has_dup_obs = not adata.obs_names.is_unique
        adata.obs, obs_col_mapping = standardize_obs(
            adata.obs, dataset_name=dataset_name, make_unique=has_dup_obs
        )
        if obs_col_mapping:
            source_meta["obs_columns_standardized"] = obs_col_mapping
        if has_dup_obs:
            source_meta["obs_name_strategy"] = "make_unique"

        # -- 9. Standardize var ------------------------------------------------
        adata.var, rename_map = standardize_var(adata.var, return_rename_map=True)
        if rename_map:
            source_meta["var_name_strategy"] = "make_unique"
            source_meta["duplicate_genes_resolved"] = len(rename_map)

        # -- 10. Standardize obsm ----------------------------------------------
        if adata.obsm:
            old_obsm_keys = set(adata.obsm.keys())
            new_obsm = standardize_obsm(dict(adata.obsm))
            # Clear and re-add
            for key in list(adata.obsm.keys()):
                del adata.obsm[key]
            for key, val in new_obsm.items():
                adata.obsm[key] = val
            new_obsm_keys = set(new_obsm.keys())
            source_meta["embeddings_preserved"] = sorted(new_obsm_keys)
            source_meta["embeddings_dropped"] = sorted(old_obsm_keys - new_obsm_keys)

        # -- 11. Assign layers -------------------------------------------------
        source_layers = dict(adata.layers)
        adata.layers.clear()
        adata = assign_layers(adata, matrix_type, source_layers)

        # -- 12. Ensure sparse -------------------------------------------------
        ensure_sparse(adata)

        # -- 13. Add provenance ------------------------------------------------
        decisions_list = [f"{k}={v}" for k, v in all_decisions.items()]
        adata = add_provenance(
            adata,
            source_meta=source_meta,
            matrix_classification=matrix_type,
            decisions=decisions_list,
            all_warnings=all_warnings,
        )

        # -- 14. Validate ------------------------------------------------------
        vr = validate_adata(adata, matrix_type=matrix_type)
        if not vr.passed:
            error_json = json.dumps(
                {"status": "error", "fatal_errors": vr.fatal_errors},
                indent=2,
            )
            print(error_json, file=sys.stdout)
            return 1

        if vr.warnings:
            all_warnings.extend(vr.warnings)

        # -- 15. Generate report and audit log ---------------------------------
        n_cells_after = adata.n_obs
        n_genes_after = adata.n_vars

        report_json = generate_report_json(
            source_path=args.input,
            source_format=fmt,
            source_meta=source_meta,
            matrix_classification=classification,
            n_cells_before=n_cells_before,
            n_genes_before=n_genes_before,
            n_cells_after=n_cells_after,
            n_genes_after=n_genes_after,
            decisions=all_decisions,
            warnings=all_warnings,
        )

        x_contents = adata.uns.get("stanobj", {}).get("x_contents", matrix_type)
        audit_log = generate_audit_log(
            source_path=args.input,
            source_format=fmt,
            output_path=args.output,
            source_meta=source_meta,
            matrix_classification=classification,
            x_contents=x_contents,
            n_cells=n_cells_after,
            n_genes=n_genes_after,
            decisions=all_decisions,
            warnings=all_warnings,
        )

        # -- 16. Write outputs -------------------------------------------------
        write_outputs(adata, args.output, report_json, audit_log)

        # -- 17. Success -------------------------------------------------------
        return 0

    except DecisionNeeded as dn:
        print(dn.to_json(), file=sys.stdout)
        return 10

    except Exception as exc:
        error_json = json.dumps(
            {"status": "error", "message": str(exc)}, indent=2
        )
        print(error_json, file=sys.stderr)
        return 1

    finally:
        for tp in temp_paths:
            cleanup_temp(tp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
