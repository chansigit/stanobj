#!/usr/bin/env python3
"""
stanobj PostToolUse hook — family handoff to stangene.

Triggers after a Bash tool call. If the command was a successful invocation
of ``stanobj`` (entry point, ``python stanobj.py``, or ``python -m
scripts.stanobj``) and exited 0, inject a handoff hint so the main agent
knows the natural next step in the stan* family pipeline is ``stangene``.

Design rules:
- Never block. This is PostToolUse — the tool already executed.
- No-op on any non-stanobj command or non-zero exit, silently.
- No-op on any parse error or missing field (hook must never break
  a working session).
- Exit 0 in all cases. Feedback is delivered via JSON stdout
  (``hookSpecificOutput.additionalContext``), with stderr as a
  best-effort fallback for older runtimes.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional

# Match:
#   - `stanobj <arg>` (pyproject entry point) — at a shell boundary, followed by space/EOL
#   - `python[3] [-m] [<path-or-module>]stanobj[.py|.sub] ...`
# Dotted module paths (e.g. `python -m scripts.stanobj`) are supported.
# Not matched: `stanobj-notes.log`, `stanobj.zarr`, `cat path/to/stanobj.txt`.
# Package-manager commands (`pip install stanobj`, etc.) are filtered out by
# `_PKG_MGR_CMD` below.
_STANOBJ_CMD = re.compile(
    r"(?:^|[\s;|&\n])"
    r"(?:"
    r"stanobj(?:\s|$)"
    r"|python\d?\s+(?:-m\s+)?[\w./-]*\bstanobj\b"
    r")"
)

# Silence false positives from package-manager invocations. If the shell
# command is installing / uninstalling / querying a package, we do not want
# to emit a handoff hint — the user is managing their environment, not
# running the tool.
_PKG_MGR_CMD = re.compile(
    r"\b(?:pip|pip3|conda|mamba|micromamba|uv|poetry|pipx)"
    r"\s+(?:install|uninstall|add|remove|show|list|search|info|update|upgrade|sync)\b"
)

_HANDOFF = (
    "✅ stanobj 执行成功。在 stan* 家族流水线里,stanobj 的下游是 **stangene**。\n"
    "\n"
    "stangene 消费 stanobj 产出的标准 h5ad,对 gene ID / symbol / alias 做 harmonization,\n"
    "为跨数据集对齐准备一致的 var 索引。\n"
    "\n"
    "建议的下一步:\n"
    "  • 如果当前在 eca-curation pipeline session 里:运行\n"
    "      /eca-run <dataset>\n"
    "    推进到 01_gene_harmonize 阶段(stage script 会调 stangene)。\n"
    "  • 否则直接告知用户:“接下来用 stangene 做 gene harmonization”,\n"
    "    让 stangene skill 接管。\n"
    "\n"
    "跳过 stangene 的合理情形:var 索引已是干净的 canonical ID,或用户显式表示跳过 "
    "gene harmonization。其余情况默认应继续到 stangene。"
)


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _get_bash_command(payload: dict) -> Optional[str]:
    tool_name = (
        payload.get("tool_name")
        or (payload.get("tool_use") or {}).get("name")
        or (payload.get("toolUse") or {}).get("name")
    )
    if tool_name != "Bash":
        return None
    tool_input = (
        payload.get("tool_input")
        or (payload.get("tool_use") or {}).get("input")
        or (payload.get("toolUse") or {}).get("input")
        or {}
    )
    cmd = tool_input.get("command")
    return cmd if isinstance(cmd, str) else None


def _exit_code(payload: dict) -> int:
    res = (
        payload.get("tool_response")
        or payload.get("tool_result")
        or payload.get("toolResult")
        or {}
    )
    if not isinstance(res, dict):
        return 0
    for key in ("exit_code", "exitCode", "returncode", "returnCode"):
        code = res.get(key)
        if code is not None:
            try:
                return int(code)
            except (TypeError, ValueError):
                return 1
    if res.get("is_error") or res.get("isError"):
        return 1
    return 0


def main() -> int:
    payload = _read_payload()
    cmd = _get_bash_command(payload)
    if not cmd:
        return 0
    if _exit_code(payload) != 0:
        return 0

    # Split by shell operators so chained commands like
    # ``pip install stanobj && stanobj in -o out`` still trigger on the run
    # segment while the install segment is skipped.
    matched = False
    for seg in re.split(r"(?:&&|\|\||;|\|)", cmd):
        if _PKG_MGR_CMD.search(seg):
            continue
        if _STANOBJ_CMD.search(seg):
            matched = True
            break
    if not matched:
        return 0

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": _HANDOFF,
        }
    }
    try:
        print(json.dumps(out))
    except Exception:
        pass
    try:
        print(_HANDOFF, file=sys.stderr)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
