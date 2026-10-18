#!/usr/bin/env python3
"""
record_environment.py
======================

Writes ``artifacts/deployment/environment.txt``: a timestamped snapshot of
exactly what produced the current build — the Python interpreter, every
installed Python package (``pip freeze``), and the Node/npm versions used by
the frontend. Generated, not hand-typed, so it can't drift from what's
actually installed the way a manually-edited note would.

Usage
-----
    python scripts/record_environment.py

Run this from the same environment (venv, or inside the backend Docker
image) that you want a record of. Node/npm are captured on a best-effort
basis: their absence from PATH (e.g. inside the backend-only container,
which has no Node toolchain) is reported in the file rather than failing
the script.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from app.core.paths import ARTIFACTS_DIR  # noqa: E402

ENVIRONMENT_TXT = ARTIFACTS_DIR / "deployment" / "environment.txt"


def _run(cmd: list[str]) -> str | None:
    """Run ``cmd`` and return its stdout, or None if it can't be run at all."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def build_environment_report() -> str:
    lines: list[str] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    lines.append(f"# Environment snapshot — generated {generated_at}")
    lines.append("# by scripts/record_environment.py — do not hand-edit, re-run instead.")
    lines.append("")

    lines.append("## Python")
    lines.append(f"python: {sys.version.replace(chr(10), ' ')}")
    python_version_cmd = _run([sys.executable, "--version"])
    if python_version_cmd:
        lines.append(f"python --version: {python_version_cmd}")
    lines.append("")

    lines.append("## Node / npm (frontend toolchain)")
    node_version = _run(["node", "--version"])
    npm_version = _run(["npm", "--version"])
    lines.append(f"node --version: {node_version if node_version else 'not found on PATH'}")
    lines.append(f"npm --version: {npm_version if npm_version else 'not found on PATH'}")
    lines.append("")

    lines.append("## pip freeze")
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    if pip_freeze:
        lines.append(pip_freeze)
    else:
        lines.append("(pip freeze failed or returned nothing)")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ENVIRONMENT_TXT.parent.mkdir(parents=True, exist_ok=True)
    report = build_environment_report()
    ENVIRONMENT_TXT.write_text(report)
    print(f"Wrote {ENVIRONMENT_TXT}")


if __name__ == "__main__":
    main()
