"""Every bug ID in the ledger (specs/README.md) must have a regression test named test_<ID>_*."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ledger_ids(markdown: str) -> list[str]:
    return re.findall(r"^\| ([A-Z]+-\d+) \|", markdown, re.M)


def collect_test_names() -> set[str]:
    names = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


def missing_ids(ids, names) -> list[str]:
    # trailing underscore: test_GC_1_ must not be satisfied by test_GC_10_
    return [i for i in ids if not any(n.startswith(f"test_{i.replace('-', '_')}_") for n in names)]


def test_every_ledger_id_has_a_regression_test():
    ids = ledger_ids((ROOT / "specs" / "README.md").read_text())
    assert len(ids) == 29
    assert missing_ids(ids, collect_test_names()) == []


def test_meta_check_catches_an_untested_id():
    names = {"test_GC_10_truncated_tail", "test_GC_1_exact_keys"}
    assert missing_ids(["GC-1", "GC-10", "GC-12"], names) == ["GC-12"]
    assert missing_ids(["GC-1"], {"test_GC_10_x"}) == ["GC-1"]
