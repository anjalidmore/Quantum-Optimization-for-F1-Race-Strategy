"""
The XGBoost / PyTorch OpenMP conflict, and the guard that prevents it.

Task 6 uses XGBoost; Task 7 uses Keras on the PyTorch backend. On macOS both
ship their own copy of ``libomp.dylib`` (scikit-learn ships a third), and
whichever copy is loaded **first** claims the process. XGBoost then crashes
with a hard segmentation fault — not an exception, a SIGSEGV that takes the
interpreter down — if it has to initialise against PyTorch's copy.

This is not theoretical: it took down the whole test suite with
``Fatal Python error: Segmentation fault`` when torch was first added, and it
is invisible to normal testing because the process dies before any assertion
runs.

``app.core.runtime.prepare_dl_runtime()`` fixes it by importing XGBoost first.
The tests below run in **subprocesses**, because a test that reproduces a
segfault in-process would kill the test runner.

Note: ``KMP_DUPLICATE_LIB_OK=TRUE``, the usual folk remedy, does **not** help
here — verified, and asserted below so nobody wastes time reaching for it.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

XGBOOST_FIT = "XGBClassifier(n_estimators=5).fit(np.random.rand(50, 3), (np.random.rand(50) > .5).astype(int))"


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )


def _xgboost_installed() -> bool:
    return _run("import xgboost").returncode == 0


requires_xgboost = pytest.mark.skipif(
    not _xgboost_installed(), reason="xgboost not installed; there is no OpenMP conflict to guard"
)


@requires_xgboost
def test_guarded_import_order_lets_xgboost_and_keras_coexist():
    """The path production code actually takes must work."""
    result = _run(f"""
        from app.core.runtime import prepare_dl_runtime
        prepare_dl_runtime()
        import keras  # noqa: F401
        import numpy as np
        from xgboost import XGBClassifier
        {XGBOOST_FIT}
        print("OK")
    """)
    assert result.returncode == 0, (
        f"the guarded order crashed (exit {result.returncode}):\n{result.stderr[-2000:]}"
    )
    assert "OK" in result.stdout


@requires_xgboost
def test_prepare_dl_runtime_reports_that_it_preloaded_xgboost():
    """The guard must actually do the thing it claims, not just set a backend."""
    result = _run("""
        from app.core.runtime import prepare_dl_runtime
        info = prepare_dl_runtime()
        assert info["xgboost_preloaded"] is True, info
        assert info["backend"] == "torch", info
        import sys
        # xgboost must already be in sys.modules before keras is imported.
        assert "xgboost" in sys.modules
        assert "keras" not in sys.modules
        print("OK")
    """)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


@requires_xgboost
def test_prepare_dl_runtime_is_idempotent():
    result = _run("""
        from app.core.runtime import prepare_dl_runtime
        first = prepare_dl_runtime()
        second = prepare_dl_runtime()
        assert first["already"] is False and second["already"] is True, (first, second)
        print("OK")
    """)
    assert result.returncode == 0, result.stderr[-2000:]


@requires_xgboost
def test_every_dl_entry_point_goes_through_the_guard():
    """Importing any DL or XAI module must be enough — a caller should not have
    to remember to call prepare_dl_runtime() itself."""
    for module in (
        "app.intelligence.dl.models",
        "app.intelligence.dl.training",
        "app.intelligence.dl.persistence",
        "app.intelligence.dl.pipeline",
        "app.intelligence.xai.loading",
        "app.intelligence.xai.pipeline",
    ):
        result = _run(f"""
            import sys
            import {module}  # noqa: F401
            assert "xgboost" in sys.modules, (
                "{module} pulled in keras/torch without claiming OpenMP for xgboost first"
            )
            import numpy as np
            from xgboost import XGBClassifier
            {XGBOOST_FIT}
            print("OK")
        """)
        assert result.returncode == 0, (
            f"importing {module} left xgboost unable to fit (exit {result.returncode}):\n"
            f"{result.stderr[-1500:]}"
        )


@requires_xgboost
@pytest.mark.slow
def test_the_unguarded_order_really_does_crash():
    """Documents the failure this guard exists for.

    If this ever stops crashing — a fixed torch, a fixed xgboost, a different
    platform — that is worth knowing, because the guard could then be
    simplified. It is asserted as "crashes OR is fine", with the outcome
    surfaced, rather than as a hard requirement that the bug still exists.
    """
    result = _run(f"""
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        import keras  # noqa: F401  -- claims OpenMP for torch
        import numpy as np
        from xgboost import XGBClassifier
        {XGBOOST_FIT}
        print("NO CRASH")
    """)
    if result.returncode == 0:
        pytest.skip(
            "The unguarded order no longer crashes on this platform/versions. "
            "prepare_dl_runtime() is now belt-and-braces rather than load-bearing; "
            "re-evaluate whether it can be simplified."
        )
    # A segfault shows up as a negative return code (-11) or 139 depending on shell.
    assert result.returncode in (-11, 139, 134, -6), (
        f"the unguarded order failed in an unexpected way (exit {result.returncode}): "
        f"{result.stderr[-500:]}"
    )


@requires_xgboost
@pytest.mark.slow
def test_kmp_duplicate_lib_ok_does_not_fix_it():
    """The usual folk remedy for duplicate OpenMP runtimes. It does not work
    here, and it is documented as unsafe besides — asserted so nobody reaches
    for it instead of the import-order fix."""
    result = _run(f"""
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        import keras  # noqa: F401
        import numpy as np
        from xgboost import XGBClassifier
        {XGBOOST_FIT}
        print("NO CRASH")
    """, env_extra={"KMP_DUPLICATE_LIB_OK": "TRUE"})
    if result.returncode == 0:
        pytest.skip("KMP_DUPLICATE_LIB_OK appears to work on this platform; re-evaluate the guard.")
    assert result.returncode != 0
