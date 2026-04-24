"""Python bridge that invokes R scripts (seurat_reader.R, sce_reader.R) via subprocess."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .base import DecisionNeeded, ReaderResult
    from .mtx_reader import read_mtx
except ImportError:
    from base import DecisionNeeded, ReaderResult
    from mtx_reader import read_mtx

# Directory where the R scripts live (next to this Python file)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_rscript() -> str:
    """Return the Rscript executable path, or raise RuntimeError."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is not available on this system. "
            "Please install R (>= 4.0) and the required packages "
            "(Seurat, SingleCellExperiment, Matrix, jsonlite) to read "
            "RDS files directly. Alternatively, convert your file in R:\n\n"
            "  library(Seurat)\n"
            "  obj <- readRDS('your_file.rds')\n"
            "  library(SeuratDisk)\n"
            "  SaveH5Seurat(obj, 'converted.h5seurat')\n"
            "  Convert('converted.h5seurat', dest='h5ad')\n"
        )
    return rscript


def _parse_r_json(stdout: str) -> dict:
    """Extract the last JSON object from R stdout (ignore library messages)."""
    # R may print warnings/messages before the JSON; find the JSON block.
    # The JSON block starts with '{' and ends with '}'.
    lines = stdout.strip().splitlines()
    json_start = None
    json_end = None
    brace_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if json_start is None and stripped.startswith("{"):
            json_start = i
            brace_depth = 0
        if json_start is not None:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth == 0:
                json_end = i
                # Don't break -- we want the LAST complete JSON block

    if json_start is None:
        raise RuntimeError(f"No JSON found in R output:\n{stdout}")

    # Re-scan from the end to find the last JSON block
    json_end = None
    json_start = None
    brace_depth = 0
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if json_end is None and stripped.endswith("}"):
            json_end = i
            brace_depth = 0
        if json_end is not None:
            brace_depth += stripped.count("}") - stripped.count("{")
            if brace_depth == 0:
                json_start = i
                break

    if json_start is None or json_end is None:
        raise RuntimeError(f"Could not parse JSON from R output:\n{stdout}")

    json_text = "\n".join(lines[json_start : json_end + 1])
    return json.loads(json_text)


def _merge_metadata(adata, metadata_path: str) -> None:
    """Merge metadata.csv exported by R into adata.obs.

    Alignment strategy, in order:
      1. Full name-based: CSV index == adata.obs index as a set → reindex.
      2. Partial name-based: non-empty intersection → assign only overlap.
      3. Positional fallback: no name overlap but row counts match.
    """
    if not os.path.isfile(metadata_path):
        return
    # Read with string index so barcode comparisons always work.
    meta = pd.read_csv(metadata_path, index_col=0)
    meta.index = meta.index.astype(str)

    obs_index = adata.obs.index.astype(str)
    common = obs_index.intersection(meta.index)

    if len(common) == len(obs_index):
        aligned = meta.reindex(obs_index)
        for col in aligned.columns:
            adata.obs[col] = aligned[col].values
    elif len(common) > 0:
        aligned = meta.loc[common]
        for col in aligned.columns:
            adata.obs.loc[common, col] = aligned[col].values
    elif len(meta) == adata.n_obs:
        # Fall back to positional only when names share nothing.
        for col in meta.columns:
            adata.obs[col] = meta[col].values


def _merge_reductions(adata, export_dir: str) -> list[str]:
    """Merge reduction_*.csv files into adata.obsm. Return names merged.

    Rows are realigned to adata.obs_names by name when the CSV rownames
    cover every cell. Partial coverage raises (silent NaN padding would
    corrupt downstream plotting). When the CSV has no name overlap but
    row counts match, fall back to positional assignment.
    """
    merged = []
    obs_index = adata.obs.index.astype(str)
    for fname in sorted(os.listdir(export_dir)):
        if not (fname.startswith("reduction_") and fname.endswith(".csv")):
            continue
        rname = fname[len("reduction_") : -len(".csv")]
        rpath = os.path.join(export_dir, fname)
        df = pd.read_csv(rpath, index_col=0)
        df.index = df.index.astype(str)
        key = f"X_{rname}"

        common = obs_index.intersection(df.index)
        if len(common) == len(obs_index):
            values = df.reindex(obs_index).values
        elif len(common) > 0:
            raise ValueError(
                f"reduction_{rname}.csv only covers {len(common)}/"
                f"{len(obs_index)} cells; cannot align embedding without "
                f"losing data."
            )
        elif len(df) == len(obs_index):
            values = df.values
        else:
            raise ValueError(
                f"reduction_{rname}.csv has {len(df)} rows but adata has "
                f"{len(obs_index)} cells, and no names overlap."
            )
        adata.obsm[key] = values.astype(np.float32)
        merged.append(rname)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_seurat(
    path: str, decisions: Optional[dict] = None
) -> ReaderResult:
    """Read a Seurat RDS file by invoking R via subprocess.

    Parameters
    ----------
    path : str
        Path to the ``.rds`` file.
    decisions : dict, optional
        May contain ``"assay"`` key to select a specific assay.

    Returns
    -------
    ReaderResult
    """
    rscript = _find_rscript()
    r_script = os.path.join(_SCRIPTS_DIR, "seurat_reader.R")

    tmp_dir = tempfile.mkdtemp(prefix="stanobj_seurat_")
    try:
        cmd = [rscript, r_script, os.path.abspath(path), tmp_dir]
        assay = (decisions or {}).get("assay_selection") or (decisions or {}).get("assay")
        if assay:
            cmd += ["--assay", assay]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Parse JSON from stdout
        try:
            info = _parse_r_json(proc.stdout)
        except RuntimeError:
            # If we can't parse JSON, report the raw output
            raise RuntimeError(
                f"Seurat reader failed (exit {proc.returncode}).\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        status = info.get("status", "error")

        # If the object was actually a SingleCellExperiment, redirect
        if status == "error":
            detected = info.get("detected_class", "")
            if "SingleCellExperiment" in detected:
                return read_sce(path, decisions=decisions)
            raise RuntimeError(info.get("message", "Unknown R error"))

        if status == "decision_needed":
            raise DecisionNeeded(
                decision_type=info.get("decision_type", "assay_selection"),
                context=f"Seurat object at {path} has multiple assays",
                options=info.get("options", []),
                recommendation=info.get("recommendation"),
                reason="Multiple assays found; please select one.",
            )

        # status == "success"
        # Read the exported MTX directory using mtx_reader
        mtx_result = read_mtx(tmp_dir)
        adata = mtx_result.adata

        # Merge cell metadata
        _merge_metadata(adata, os.path.join(tmp_dir, "metadata.csv"))

        # Merge reductions
        reductions = _merge_reductions(adata, tmp_dir)

        # Map slot_used -> matrix_type_hint
        slot_used = info.get("slot_used", "counts")
        slot_map = {
            "counts": "counts",
            "data": "log1p",
            "scale.data": "scaled",
        }
        matrix_type_hint = slot_map.get(slot_used, "unknown")

        # Build source_meta, merging mtx_reader metadata as base
        source_meta = dict(mtx_result.source_meta)
        source_meta.update({
            "source_format": "seurat_rds",
            "reader_used": "r_bridge/seurat",
            "matrix_orientation_before": "genes_x_cells",
            "transposed": True,
            "raw_counts_found": slot_used == "counts",
            "feature_types_present": source_meta.get("feature_types_present", []),
            "matrix_type_hint": matrix_type_hint,
            "decompressed": False,
            "warnings": source_meta.get("warnings", []),
            "assay": info.get("assay"),
            "slot_used": slot_used,
            "n_cells": info.get("n_cells"),
            "n_features": info.get("n_features"),
            "reductions_exported": reductions,
        })

        return ReaderResult(adata=adata, source_meta=source_meta)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def read_sce(
    path: str, decisions: Optional[dict] = None
) -> ReaderResult:
    """Read a SingleCellExperiment RDS file by invoking R via subprocess.

    Parameters
    ----------
    path : str
        Path to the ``.rds`` file.
    decisions : dict, optional
        May contain ``"assay"`` key to select a specific assay.

    Returns
    -------
    ReaderResult
    """
    rscript = _find_rscript()
    r_script = os.path.join(_SCRIPTS_DIR, "sce_reader.R")

    tmp_dir = tempfile.mkdtemp(prefix="stanobj_sce_")
    try:
        cmd = [rscript, r_script, os.path.abspath(path), tmp_dir]
        assay = (decisions or {}).get("assay_selection") or (decisions or {}).get("assay")
        if assay:
            cmd += ["--assay", assay]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Parse JSON from stdout
        try:
            info = _parse_r_json(proc.stdout)
        except RuntimeError:
            raise RuntimeError(
                f"SCE reader failed (exit {proc.returncode}).\n"
                f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
            )

        status = info.get("status", "error")

        if status == "error":
            raise RuntimeError(info.get("message", "Unknown R error"))

        if status == "decision_needed":
            raise DecisionNeeded(
                decision_type=info.get("decision_type", "assay_selection"),
                context=f"SingleCellExperiment at {path} has multiple assays",
                options=info.get("options", []),
                recommendation=info.get("recommendation"),
                reason="Multiple assays found; please select one.",
            )

        # status == "success"
        mtx_result = read_mtx(tmp_dir)
        adata = mtx_result.adata

        # Merge cell metadata
        _merge_metadata(adata, os.path.join(tmp_dir, "metadata.csv"))

        # Merge reductions
        reductions = _merge_reductions(adata, tmp_dir)

        # Map slot_hint -> matrix_type_hint
        slot_hint = info.get("slot_hint", "counts")
        hint_map = {
            "counts": "counts",
            "logcounts": "log1p",
            "normcounts": "normalized",
        }
        matrix_type_hint = hint_map.get(slot_hint, "unknown")

        # Build source_meta, merging mtx_reader metadata as base
        source_meta = dict(mtx_result.source_meta)
        source_meta.update({
            "source_format": "sce",
            "reader_used": "r_bridge/sce",
            "matrix_orientation_before": "genes_x_cells",
            "transposed": True,
            "raw_counts_found": slot_hint == "counts",
            "feature_types_present": source_meta.get("feature_types_present", []),
            "matrix_type_hint": matrix_type_hint,
            "decompressed": False,
            "warnings": source_meta.get("warnings", []),
            "assay": info.get("assay"),
            "slot_hint": slot_hint,
            "n_cells": info.get("n_cells"),
            "n_features": info.get("n_features"),
            "reductions_exported": reductions,
            "colData_cols": info.get("colData_cols", []),
            "rowData_cols": info.get("rowData_cols", []),
        })

        return ReaderResult(adata=adata, source_meta=source_meta)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
