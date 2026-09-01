"""Repo-wide guards that documentation and runtime compatibility do not rot.

Two things this defends, both of which have already gone wrong once:

1. PYTHON 3.9 COMPATIBILITY of the scheduled scripts. The launchd agents that
   run the live prospective cohort twice a day invoke `/usr/bin/python3`, which
   on this macOS is 3.9.6 -- not the project's .venv. A `str | None` annotation
   or a `match` statement in those files is a silent production break: the
   tracker dies at 09:05 and the day's snapshot is lost. This test compiles
   them with the real 3.9 interpreter when it exists, and falls back to an AST
   check for modern syntax when it does not.

2. PUBLIC API DOCUMENTATION. Every function whose name does not start with "_"
   is something another module (or a teammate) is expected to call, so it needs
   a docstring and a typed signature. Private helpers are exempt: they are read
   in context.

Deliberate exemptions live in EXEMPT below, each with a reason. An exemption is
a decision, not a way to silence the check -- if the list grows, that is the
signal to fix the code rather than extend the list.

    .venv/bin/python tests/test_docs_and_compat.py
"""
import ast
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Executed by launchd with /usr/bin/python3 (3.9.6), NOT the project venv.
SCHEDULED_39 = [
    "02_Data/track_new_videos.py",
    "02_Data/yt_shorts.py",           # imported by the tracker
    "02_Data/backfill_published_at.py",
]

# Everything whose public surface is checked. Tests and one-off scans excluded.
CHECKED = [
    "02_Data/track_new_videos.py",
    "02_Data/yt_shorts.py",
    "02_Data/backfill_published_at.py",
    "02_Data/clean_retrospective.py",
    "02_Data/compute_labels_v2.py",
    "02_Data/eda_retrospective.py",
    "02_Data/collect_and_extract.py",
    "03_Models/ytdiag/adapters.py",
    "03_Models/ytdiag/features.py",
    "03_Models/ytdiag/split.py",
    "03_Models/ytdiag/baselines.py",
    "03_Models/ytdiag/synthetic.py",
    "03_Models/run_baselines.py",
]

EXEMPT = {
    # module-level CLI wiring: argparse describes itself, and __doc__ is the help text
    ("02_Data/track_new_videos.py", "main"),
    ("02_Data/clean_retrospective.py", "main"),
    ("02_Data/compute_labels_v2.py", "main"),
    ("02_Data/eda_retrospective.py", "main"),
    ("02_Data/backfill_published_at.py", "main"),
    ("02_Data/collect_and_extract.py", "main"),
    ("03_Models/run_baselines.py", "main"),
}


def _public_functions(path):
    """(name, node) for every module-level public function in `path`.

    Module level only: a nested closure is an implementation detail of the
    function that owns it, not part of the module's surface."""
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    return [(n.name, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]


def _is_annotated(node):
    """A signature counts as annotated when it declares a return type and every
    parameter that is not self/cls/*args/**kwargs carries a type."""
    params = [a for a in node.args.args if a.arg not in ("self", "cls")]
    if node.returns is None:
        return False
    return all(a.annotation is not None for a in params)


def test_public_functions_have_docstrings():
    missing = []
    for path in CHECKED:
        for name, node in _public_functions(path):
            if (path, name) in EXEMPT:
                continue
            if not ast.get_docstring(node):
                missing.append(f"{path}:{node.lineno} {name}()")
    assert not missing, (
        f"{len(missing)} public functions without a docstring:\n  " + "\n  ".join(missing)
        + "\n\nWrite what a caller cannot infer: units, what None means, why the "
          "constraint exists. Or add it to EXEMPT with a reason.")


def test_public_functions_are_type_annotated():
    missing = []
    for path in CHECKED:
        for name, node in _public_functions(path):
            if (path, name) in EXEMPT:
                continue
            if not _is_annotated(node):
                missing.append(f"{path}:{node.lineno} {name}()")
    assert not missing, (
        f"{len(missing)} public functions with an unannotated signature:\n  " + "\n  ".join(missing)
        + "\n\nAdd `from __future__ import annotations` at the top of the file so the "
          "annotations stay python3.9-safe, then annotate params and the return type.")


def test_scheduled_scripts_stay_python39_compatible():
    """The launchd agents run these with /usr/bin/python3 (3.9.6) against the
    live cohort. Modern syntax here is a silent production break, not a lint."""
    py39 = "/usr/bin/python3"
    real_39 = False
    if os.path.exists(py39):
        v = subprocess.run([py39, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True)
        real_39 = v.stdout.strip().startswith("3.9")

    for path in SCHEDULED_39:
        full = os.path.join(ROOT, path)
        if real_39:
            r = subprocess.run([py39, "-c", f"import ast; ast.parse(open({full!r}).read())"],
                               capture_output=True, text=True)
            assert r.returncode == 0, (
                f"{path} does not parse under {py39} (the launchd interpreter):\n{r.stderr}")
        src = open(full, encoding="utf-8").read()
        tree = ast.parse(src)
        # `X | Y` and `list[str]` are fine INSIDE annotations once the file has
        # `from __future__ import annotations` (they are never evaluated), but a
        # runtime use crashes on 3.9.
        has_future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                         and any(a.name == "annotations" for a in n.names) for n in tree.body)
        annotated = any(_is_annotated(n) for n in tree.body if isinstance(n, ast.FunctionDef))
        assert has_future or not annotated, (
            f"{path} has annotated signatures but no `from __future__ import annotations`. "
            f"Without it, a `str | None` annotation is evaluated at import time and "
            f"raises TypeError on python3.9 -- killing the scheduled run.")
        for node in ast.walk(tree):
            assert not isinstance(node, ast.Match), (
                f"{path}: `match` statement is python3.10+, but this file runs on 3.9")


def test_every_checked_file_exists():
    """Guards the guard: a renamed file must not silently drop out of coverage."""
    for path in CHECKED + SCHEDULED_39:
        assert os.path.exists(os.path.join(ROOT, path)), f"{path} in the check list does not exist"


def test_exemptions_are_still_real():
    """An exemption for a function that no longer exists is stale bookkeeping."""
    stale = []
    for path, name in EXEMPT:
        if not os.path.exists(os.path.join(ROOT, path)):
            stale.append(f"{path} (file gone)")
            continue
        if name not in {n for n, _ in _public_functions(path)}:
            stale.append(f"{path}:{name}")
    assert not stale, "stale entries in EXEMPT:\n  " + "\n  ".join(stale)


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"{name} OK")
            except AssertionError as e:
                failures += 1
                print(f"{name} FAILED\n{e}\n")
    if failures:
        sys.exit(f"{failures} check(s) failed")
    print("ALL DOC + COMPAT CHECKS PASSED")
