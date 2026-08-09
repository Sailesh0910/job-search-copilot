"""
ARCHITECTURE.md makes several claims about the codebase's layering that were
previously only checked "with a plain grep" by hand. This turns those claims
into an enforced regression test: psycopg2 (the database driver) is only
ever imported in lakebase.py, requests (the HTTP client) only in
adzuna_client.py, and main.py never reaches around job_broker.py to talk to
the database directly.
"""

import ast
import os

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")


def _top_level_imports(filepath) -> set:
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _app_py_files(exclude=()):
    return [
        f for f in os.listdir(APP_DIR)
        if f.endswith(".py") and f not in exclude
    ]


def test_psycopg2_only_imported_in_lakebase():
    offenders = [
        f for f in _app_py_files(exclude={"lakebase.py"})
        if "psycopg2" in _top_level_imports(os.path.join(APP_DIR, f))
    ]
    assert not offenders, f"psycopg2 imported outside lakebase.py: {offenders}"


def test_requests_only_imported_in_adzuna_client_and_job_broker():
    """job_broker.py's one exception is draft_cover_letter, which calls the
    AI Gateway directly (no SDK wrapper covers that route today)."""
    offenders = [
        f for f in _app_py_files(exclude={"adzuna_client.py", "job_broker.py"})
        if "requests" in _top_level_imports(os.path.join(APP_DIR, f))
    ]
    assert not offenders, f"requests imported outside adzuna_client.py/job_broker.py: {offenders}"


def test_main_never_imports_lakebase_directly():
    """main.py's routes should only ever call job_broker — the whole point
    of the broker pattern is that SQL stays out of the web layer."""
    imports = _top_level_imports(os.path.join(APP_DIR, "main.py"))
    assert "lakebase" not in imports


def test_mcp_tools_never_imports_lakebase_directly():
    imports = _top_level_imports(os.path.join(APP_DIR, "mcp_tools.py"))
    assert "lakebase" not in imports
