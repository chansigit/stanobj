"""Archive extraction safety: zip-slip, tar traversal, symlink descent."""

from __future__ import annotations

import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils import cleanup_temp, extract_archive


# ---------------------------------------------------------------------------
# Tar path traversal
# ---------------------------------------------------------------------------


class TestTarTraversal:
    def test_rejects_relative_traversal(self, tmp_dir):
        """A tar entry named '../outside/evil.txt' must not be written to
        the parent of the extraction directory."""
        outside = os.path.join(tmp_dir, "outside")
        os.makedirs(outside)
        dest = os.path.join(tmp_dir, "extract")
        os.makedirs(dest)

        payload = os.path.join(tmp_dir, "payload.txt")
        with open(payload, "w") as f:
            f.write("evil")

        archive = os.path.join(tmp_dir, "malicious.tar")
        with tarfile.open(archive, "w") as tf:
            tf.add(payload, arcname="../outside/evil.txt")

        # Extract — either raises or silently skips the bad entry.
        try:
            extract_archive(archive, dest=dest)
        except Exception:
            pass

        # The evil file must NOT exist outside dest.
        assert not os.path.exists(os.path.join(outside, "evil.txt")), (
            "Tar traversal succeeded — archive escaped extraction dir"
        )

    def test_rejects_absolute_path(self, tmp_dir):
        """A tar entry with an absolute path must not write to that path."""
        payload = os.path.join(tmp_dir, "payload.txt")
        with open(payload, "w") as f:
            f.write("evil")

        absolute_target = os.path.join(tmp_dir, "absolute_escape.txt")
        archive = os.path.join(tmp_dir, "abs.tar")
        with tarfile.open(archive, "w") as tf:
            tf.add(payload, arcname=absolute_target)

        dest = os.path.join(tmp_dir, "extract_abs")
        os.makedirs(dest)

        try:
            extract_archive(archive, dest=dest)
        except Exception:
            pass

        # The absolute target outside dest must NOT have been written.
        # (payload.txt already exists at absolute_target — we check it was
        #  NOT overwritten with evil contents from the tar entry).
        # Actually the payload was already written with "evil", so instead
        # verify nothing was written to a sibling of dest.
        for entry in os.listdir(tmp_dir):
            if entry in ("abs.tar", "extract_abs", "payload.txt"):
                continue
            raise AssertionError(f"Unexpected extraction to {entry}")


# ---------------------------------------------------------------------------
# Zip path traversal
# ---------------------------------------------------------------------------


class TestZipTraversal:
    def test_rejects_relative_traversal(self, tmp_dir):
        """A zip entry named '../outside/evil.txt' must not escape dest."""
        outside = os.path.join(tmp_dir, "outside")
        os.makedirs(outside)
        dest = os.path.join(tmp_dir, "extract")
        os.makedirs(dest)

        archive = os.path.join(tmp_dir, "malicious.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../outside/evil.txt", "nope")

        try:
            extract_archive(archive, dest=dest)
        except Exception:
            pass

        assert not os.path.exists(os.path.join(outside, "evil.txt")), (
            "Zip-slip succeeded — archive escaped extraction dir"
        )


# ---------------------------------------------------------------------------
# Symlink descent
# ---------------------------------------------------------------------------


class TestSymlinkDescent:
    def test_resolve_input_does_not_follow_symlink_subdir(self, tmp_dir):
        """If an archive extracts to a single symlinked entry pointing
        outside, resolve_input must NOT descend into the symlink target."""
        target = os.path.join(tmp_dir, "secret_target")
        os.makedirs(target)
        with open(os.path.join(target, "secret.txt"), "w") as f:
            f.write("should not be opened")

        archive = os.path.join(tmp_dir, "sneaky.tar")
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo(name="innocent")
            info.type = tarfile.SYMTYPE
            info.linkname = os.path.abspath(target)
            tf.addfile(info)

        from scripts.stanobj import resolve_input

        try:
            resolved, _, temps = resolve_input(archive)
        except Exception:
            # Outright rejection is also acceptable.
            return

        try:
            abs_resolved = os.path.abspath(resolved)
            abs_target = os.path.abspath(target)
            assert abs_resolved != abs_target, (
                f"resolve_input followed symlink outside the extraction dir: "
                f"{abs_resolved}"
            )
        finally:
            for t in temps:
                cleanup_temp(t)
