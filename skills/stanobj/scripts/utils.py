"""Shared utility functions for stanobj."""

from __future__ import annotations

import gzip
import bz2
import os
import shutil
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_EXTENSIONS = {".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar", ".zip"}


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def sample_matrix(
    X: "np.ndarray | sparse.spmatrix", n_sample: int = 1000, seed: int = 42
) -> np.ndarray:
    """Sample *n_sample* rows from *X*, returning a dense ndarray.

    If *X* has fewer than *n_sample* rows the full matrix is returned (dense).
    """
    n_rows = X.shape[0]
    if n_rows <= n_sample:
        return X.toarray() if sparse.issparse(X) else np.asarray(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=n_sample, replace=False)
    Xs = X[idx]
    return Xs.toarray() if sparse.issparse(Xs) else np.asarray(Xs)


def is_integer_like(arr: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if every finite value in *arr* is within *tol* of an integer."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.all(np.abs(finite - np.round(finite)) < tol))


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def make_names_unique(
    names: List[str],
) -> Tuple[List[str], dict]:
    """Make a list of names unique by appending ``-1``, ``-2``, ... suffixes.

    Returns
    -------
    unique_names : list[str]
        The de-duplicated list (same length as *names*).
    rename_map : dict
        ``{original_index: new_name}`` for every entry that was renamed.
    """
    counts: dict[str, int] = {}
    unique: List[str] = []
    rename_map: dict[int, str] = {}

    for i, name in enumerate(names):
        if name not in counts:
            counts[name] = 0
            unique.append(name)
        else:
            counts[name] += 1
            new_name = f"{name}-{counts[name]}"
            unique.append(new_name)
            rename_map[i] = new_name

    return unique, rename_map


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_number(n: int | float) -> str:
    """Format a number with comma separators (e.g. 1,234,567)."""
    return f"{n:,}"


def format_size(size_bytes: int | float) -> str:
    """Human-readable file size (e.g. '1.23 MB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{size_bytes:.0f} {unit}"
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"  # pragma: no cover


def iso_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Path / compression helpers
# ---------------------------------------------------------------------------


def strip_compression_ext(path: str | Path) -> Tuple[str, bool]:
    """Remove a compression extension (.gz, .bz2) from *path*.

    For compound extensions like ``.tar.gz`` only ``.gz`` is stripped,
    yielding a ``.tar`` file.

    Returns
    -------
    stripped : str
        Path without the compression suffix.
    was_compressed : bool
        True if a compression extension was removed.
    """
    p = str(path)
    for ext in (".gz", ".bz2"):
        if p.endswith(ext):
            return p[: -len(ext)], True
    return p, False


def is_archive(path: str | Path) -> bool:
    """Return True if *path* looks like a supported archive file."""
    p = str(path).lower()
    return any(p.endswith(ext) for ext in ARCHIVE_EXTENSIONS)


def extract_archive(path: str | Path, dest: Optional[str | Path] = None) -> str:
    """Extract an archive and return the path to the extracted directory.

    Parameters
    ----------
    path
        Path to the archive file (.tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .zip).
    dest
        Destination directory.  If *None*, a temporary directory is created.

    Returns
    -------
    str
        Path to the directory containing the extracted files.
    """
    path = str(path)
    if dest is None:
        dest = tempfile.mkdtemp(prefix="stanobj_extract_")
    dest = str(dest)

    lower = path.lower()
    dest_real = os.path.realpath(dest)
    if lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar")):
        with tarfile.open(path) as tf:
            # filter="data" (Python 3.12+) rejects absolute paths, ".."
            # traversal, device/special files, and external symlinks.
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                # Python < 3.12: manually validate each member.
                _safe_tar_extractall(tf, dest_real)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                target = os.path.realpath(os.path.join(dest, name))
                if not (target == dest_real or target.startswith(dest_real + os.sep)):
                    raise ValueError(
                        f"Unsafe zip entry escapes extraction dir: {name!r}"
                    )
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {path}")

    return dest


def _safe_tar_extractall(tf: "tarfile.TarFile", dest_real: str) -> None:
    """Fallback for pre-3.12 tarfile: reject entries that escape *dest_real*."""
    for member in tf.getmembers():
        target = os.path.realpath(os.path.join(dest_real, member.name))
        if not (target == dest_real or target.startswith(dest_real + os.sep)):
            raise ValueError(
                f"Unsafe tar entry escapes extraction dir: {member.name!r}"
            )
        if member.issym() or member.islnk():
            link_target = os.path.realpath(
                os.path.join(os.path.dirname(target), member.linkname)
            )
            if not link_target.startswith(dest_real + os.sep):
                raise ValueError(
                    f"Unsafe tar link escapes extraction dir: "
                    f"{member.name!r} -> {member.linkname!r}"
                )
    tf.extractall(dest_real)


def decompress_to_temp(path: str | Path) -> str:
    """Decompress a .gz or .bz2 file to a temporary file and return its path.

    The caller is responsible for cleaning up (see :func:`cleanup_temp`).
    """
    path = str(path)
    lower = path.lower()

    stripped, _ = strip_compression_ext(os.path.basename(path))
    suffix = os.path.splitext(stripped)[1] or ""

    fd, tmp_path = tempfile.mkstemp(prefix="stanobj_decomp_", suffix=suffix)
    os.close(fd)

    if lower.endswith(".gz"):
        with gzip.open(path, "rb") as fin, open(tmp_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    elif lower.endswith(".bz2"):
        with bz2.open(path, "rb") as fin, open(tmp_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        raise ValueError(f"Unsupported compression: {path}")

    return tmp_path


def cleanup_temp(path: str | Path) -> None:
    """Remove a temp file or directory silently (ignore errors)."""
    path = str(path)
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
