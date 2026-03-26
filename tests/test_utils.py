"""Tests for scripts.utils."""

from __future__ import annotations

import gzip
import os
import tarfile
import tempfile
import zipfile

import numpy as np
import pytest
from scipy import sparse

from scripts.utils import (
    ARCHIVE_EXTENSIONS,
    cleanup_temp,
    decompress_to_temp,
    extract_archive,
    format_number,
    format_size,
    is_archive,
    is_integer_like,
    iso_now,
    make_names_unique,
    sample_matrix,
    strip_compression_ext,
)


# -----------------------------------------------------------------------
# sample_matrix
# -----------------------------------------------------------------------


class TestSampleMatrix:
    def test_dense_smaller_than_n(self):
        X = np.arange(30).reshape(5, 6)
        out = sample_matrix(X, n_sample=1000)
        assert isinstance(out, np.ndarray)
        assert out.shape == (5, 6)
        np.testing.assert_array_equal(out, X)

    def test_dense_sampling(self):
        X = np.arange(200).reshape(100, 2)
        out = sample_matrix(X, n_sample=10, seed=0)
        assert out.shape == (10, 2)
        # deterministic
        out2 = sample_matrix(X, n_sample=10, seed=0)
        np.testing.assert_array_equal(out, out2)

    def test_sparse_returns_dense(self):
        X = sparse.random(50, 10, density=0.3, format="csr", random_state=0)
        out = sample_matrix(X, n_sample=10)
        assert isinstance(out, np.ndarray)
        assert out.shape == (10, 10)

    def test_sparse_smaller_than_n(self):
        X = sparse.random(5, 3, density=0.5, format="csr", random_state=0)
        out = sample_matrix(X, n_sample=1000)
        assert isinstance(out, np.ndarray)
        assert out.shape == (5, 3)

    def test_n_sample_equals_rows(self):
        X = np.ones((10, 4))
        out = sample_matrix(X, n_sample=10)
        assert out.shape == (10, 4)


# -----------------------------------------------------------------------
# is_integer_like
# -----------------------------------------------------------------------


class TestIsIntegerLike:
    def test_pure_integers(self):
        assert is_integer_like(np.array([0, 1, 2, 3]))

    def test_float_integers(self):
        assert is_integer_like(np.array([1.0, 2.0, 3.0]))

    def test_not_integer(self):
        assert not is_integer_like(np.array([1.5, 2.0, 3.0]))

    def test_empty_array(self):
        assert is_integer_like(np.array([]))

    def test_near_integer(self):
        assert is_integer_like(np.array([1.0 + 1e-8, 2.0 - 1e-8]))

    def test_nan_and_inf_ignored(self):
        assert is_integer_like(np.array([1.0, np.nan, np.inf, 2.0]))

    def test_all_nan(self):
        assert is_integer_like(np.array([np.nan, np.nan]))


# -----------------------------------------------------------------------
# make_names_unique
# -----------------------------------------------------------------------


class TestMakeNamesUnique:
    def test_no_duplicates(self):
        names = ["A", "B", "C"]
        unique, rmap = make_names_unique(names)
        assert unique == ["A", "B", "C"]
        assert rmap == {}

    def test_with_duplicates(self):
        names = ["TP53", "BRCA1", "TP53", "TP53"]
        unique, rmap = make_names_unique(names)
        assert unique == ["TP53", "BRCA1", "TP53-1", "TP53-2"]
        assert rmap == {2: "TP53-1", 3: "TP53-2"}

    def test_single_duplicate(self):
        names = ["X", "X"]
        unique, rmap = make_names_unique(names)
        assert unique == ["X", "X-1"]
        assert rmap == {1: "X-1"}

    def test_empty(self):
        unique, rmap = make_names_unique([])
        assert unique == []
        assert rmap == {}


# -----------------------------------------------------------------------
# strip_compression_ext
# -----------------------------------------------------------------------


class TestStripCompressionExt:
    def test_gz(self):
        stripped, was = strip_compression_ext("data.csv.gz")
        assert stripped == "data.csv"
        assert was is True

    def test_bz2(self):
        stripped, was = strip_compression_ext("data.csv.bz2")
        assert stripped == "data.csv"
        assert was is True

    def test_tar_gz(self):
        stripped, was = strip_compression_ext("archive.tar.gz")
        assert stripped == "archive.tar"
        assert was is True

    def test_no_compression(self):
        stripped, was = strip_compression_ext("data.csv")
        assert stripped == "data.csv"
        assert was is False

    def test_pathlib(self):
        from pathlib import Path

        stripped, was = strip_compression_ext(Path("foo/bar.h5ad.gz"))
        assert stripped == "foo/bar.h5ad"
        assert was is True


# -----------------------------------------------------------------------
# is_archive
# -----------------------------------------------------------------------


class TestIsArchive:
    def test_tar_gz(self):
        assert is_archive("data.tar.gz") is True

    def test_tgz(self):
        assert is_archive("data.tgz") is True

    def test_zip(self):
        assert is_archive("data.zip") is True

    def test_tar(self):
        assert is_archive("data.tar") is True

    def test_tar_bz2(self):
        assert is_archive("data.tar.bz2") is True

    def test_tbz2(self):
        assert is_archive("data.tbz2") is True

    def test_not_archive(self):
        assert is_archive("data.csv") is False
        assert is_archive("data.h5ad") is False

    def test_case_insensitive(self):
        assert is_archive("DATA.TAR.GZ") is True
        assert is_archive("DATA.ZIP") is True


# -----------------------------------------------------------------------
# extract_archive
# -----------------------------------------------------------------------


class TestExtractArchive:
    def test_tar_gz(self, tmp_dir):
        # Create a tar.gz with a single text file
        payload = os.path.join(tmp_dir, "hello.txt")
        with open(payload, "w") as f:
            f.write("hello world")

        archive = os.path.join(tmp_dir, "test.tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname="hello.txt")

        dest = os.path.join(tmp_dir, "out")
        result = extract_archive(archive, dest=dest)
        assert result == dest
        assert os.path.isfile(os.path.join(dest, "hello.txt"))

    def test_zip(self, tmp_dir):
        payload = os.path.join(tmp_dir, "hello.txt")
        with open(payload, "w") as f:
            f.write("hello zip")

        archive = os.path.join(tmp_dir, "test.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(payload, arcname="hello.txt")

        dest = os.path.join(tmp_dir, "out_zip")
        result = extract_archive(archive, dest=dest)
        assert result == dest
        assert os.path.isfile(os.path.join(dest, "hello.txt"))

    def test_auto_temp_dir(self, tmp_dir):
        payload = os.path.join(tmp_dir, "data.txt")
        with open(payload, "w") as f:
            f.write("temp test")

        archive = os.path.join(tmp_dir, "auto.tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname="data.txt")

        result = extract_archive(archive)
        try:
            assert os.path.isdir(result)
            assert os.path.isfile(os.path.join(result, "data.txt"))
        finally:
            cleanup_temp(result)

    def test_unsupported_format(self, tmp_dir):
        fake = os.path.join(tmp_dir, "data.rar")
        with open(fake, "w") as f:
            f.write("not an archive")
        with pytest.raises(ValueError, match="Unsupported archive"):
            extract_archive(fake)


# -----------------------------------------------------------------------
# decompress_to_temp
# -----------------------------------------------------------------------


class TestDecompressToTemp:
    def test_gz(self, tmp_dir):
        original = os.path.join(tmp_dir, "data.csv")
        with open(original, "w") as f:
            f.write("a,b,c\n1,2,3\n")

        compressed = os.path.join(tmp_dir, "data.csv.gz")
        with open(original, "rb") as fin, gzip.open(compressed, "wb") as fout:
            fout.write(fin.read())

        tmp_path = decompress_to_temp(compressed)
        try:
            assert os.path.isfile(tmp_path)
            with open(tmp_path) as f:
                assert f.read() == "a,b,c\n1,2,3\n"
        finally:
            cleanup_temp(tmp_path)

    def test_unsupported(self, tmp_dir):
        bad = os.path.join(tmp_dir, "data.xz")
        with open(bad, "w") as f:
            f.write("nope")
        with pytest.raises(ValueError, match="Unsupported compression"):
            decompress_to_temp(bad)


# -----------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------


class TestFormatting:
    def test_format_number(self):
        assert format_number(1234567) == "1,234,567"
        assert format_number(42) == "42"

    def test_format_size_bytes(self):
        assert format_size(500) == "500 B"

    def test_format_size_kb(self):
        assert "KB" in format_size(2048)

    def test_format_size_mb(self):
        result = format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_size_gb(self):
        result = format_size(3 * 1024 ** 3)
        assert "GB" in result

    def test_iso_now(self):
        ts = iso_now()
        # Must be a valid ISO timestamp with timezone info
        assert "T" in ts
        assert "+" in ts or "Z" in ts


# -----------------------------------------------------------------------
# cleanup_temp
# -----------------------------------------------------------------------


class TestCleanupTemp:
    def test_cleanup_file(self, tmp_dir):
        f = os.path.join(tmp_dir, "todelete.txt")
        with open(f, "w") as fh:
            fh.write("bye")
        cleanup_temp(f)
        assert not os.path.exists(f)

    def test_cleanup_dir(self, tmp_dir):
        d = os.path.join(tmp_dir, "subdir")
        os.makedirs(d)
        with open(os.path.join(d, "x.txt"), "w") as fh:
            fh.write("x")
        cleanup_temp(d)
        assert not os.path.exists(d)

    def test_cleanup_nonexistent(self):
        # Should not raise
        cleanup_temp("/tmp/nonexistent_stanobj_path_12345")
