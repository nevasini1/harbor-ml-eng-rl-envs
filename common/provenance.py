"""Where a committed number came from.

Why this exists
---------------
Every anchor in this repo is a measurement, and until now none of them recorded
when it was measured, from what code, or what it replaced. The consequences were
concrete rather than theoretical:

  * The shipping bar moved from `band_sigma >= 3.0` to a 4.0 derived from a stated
    tolerance. Nothing in a shipped `anchors.json` distinguishes an anchor screened
    under the old rule from one screened under the new one -- the only way to tell
    was to date the commit that wrote the file.
  * `band_sigma` itself meant two different things across tracks for a working day.
  * A retired criterion sat in two measurement records under a key one character
    away from the live one, undated, read by nothing.

Ten benchmark and eval projects were surveyed for prior art and not one carries a
machine-readable threshold record -- no `measured_at`, `git_commit`, `n_seeds` or
`supersedes` anywhere. The two mechanisms worth copying:

  Inspect AI's `EvalRevision`  -- `{origin, commit, dirty}`, captured from three
      subprocess calls. `dirty` is the field everyone omits and then regrets: a
      commit hash alone is a lie if the tree had uncommitted changes when the
      number was produced.
  MTEB's results JSON          -- records the code version and the dataset
      revision beside every score, so "which rule produced this" is answerable
      from the file rather than from memory.

What this deliberately does NOT do
----------------------------------
It does not invent a measurement date. `assembled_at` is when the anchors were
last written into a task tree, which is knowable; when the underlying seed runs
happened is not recoverable for records that already exist, and guessing it would
be worse than leaving it out. New measurement scripts should stamp their own
`measured_at` at the point they actually measure.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """A git query, or None where git cannot answer."""
    try:
        out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_revision() -> dict:
    """`{commit, dirty, branch}`, or `{"available": False}` outside a repo.

    Degrades rather than raising: this is used from assemblers that must still
    work in a build context or a release tarball with no `.git`.
    """
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return {"available": False}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # True means the working tree had uncommitted changes when this was
        # written, so `commit` alone does not identify the code that ran.
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def stamp(script: str, **extra: object) -> dict:
    """The provenance block to write beside a derived number.

    `script` is the file that produced it, so a reader can re-derive rather than
    trust. Callers add whatever else is knowable at the point of writing --
    `source` for the measurement record consumed, `n_seeds`, a data digest.
    """
    return {
        "assembled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": script,
        "git": git_revision(),
        **extra,
    }
