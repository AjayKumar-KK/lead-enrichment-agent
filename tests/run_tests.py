#!/usr/bin/env python3
"""Zero-dependency test runner.

``pytest tests/`` is the normal way to run this suite. This script exists so the
tests can also be run on a machine with nothing but the standard library:

    python tests/run_tests.py

It discovers ``test_*`` functions in every ``test_*.py`` module and supplies
minimal ``tmp_path`` and ``monkeypatch`` fixtures with the same semantics pytest
gives them.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class MonkeyPatch:
    """The subset of pytest's monkeypatch these tests use."""

    def __init__(self) -> None:
        self._undo: list[tuple[object, str, object, bool]] = []

    def setattr(self, target, name=None, value=None) -> None:
        if isinstance(target, str):  # "module.path.attr" form
            module_path, _, attr = target.rpartition(".")
            obj = importlib.import_module(module_path)
            name, value = attr, name
        else:
            obj = target
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        self._undo.append((obj, name, old, had))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, old, had in reversed(self._undo):
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self._undo.clear()


def build_fixtures(func):
    """Provide tmp_path / monkeypatch based on the test's signature."""
    params = inspect.signature(func).parameters
    kwargs, cleanup = {}, []

    if "tmp_path" in params:
        directory = Path(tempfile.mkdtemp(prefix="lea-test-"))
        kwargs["tmp_path"] = directory
        cleanup.append(lambda: shutil.rmtree(directory, ignore_errors=True))

    if "monkeypatch" in params:
        patcher = MonkeyPatch()
        kwargs["monkeypatch"] = patcher
        cleanup.append(patcher.undo)

    return kwargs, cleanup


def main() -> int:
    modules = sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
    passed = failed = 0
    failures: list[tuple[str, str]] = []

    for module_name in modules:
        module = importlib.import_module(f"tests.{module_name}")
        tests = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        print(f"\n{module_name}  ({len(tests)} tests)")

        for name, func in tests:
            kwargs, cleanup = build_fixtures(func)
            try:
                func(**kwargs)
                passed += 1
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                failures.append((f"{module_name}.{name}", traceback.format_exc()))
                print(f"  FAIL  {name}")
            finally:
                for undo in cleanup:
                    undo()

    print("\n" + "=" * 70)
    print(f"{passed} passed, {failed} failed")
    print("=" * 70)

    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
