import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
EXAMPLES = {p.read_text().strip() for p in (ROOT / "examples").rglob("*.py")}
PYTHON_BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)


def test_1x_package_is_gone():
    result = subprocess.run(
        [sys.executable, "-c", "import GeneOpt"], cwd=ROOT, capture_output=True, text=True
    )
    assert "ModuleNotFoundError: No module named 'GeneOpt'" in result.stderr


@pytest.mark.parametrize(
    "path", ["GeneOpt", "main.py", "main2.py", "server.py", "RunNServers.py", "setup.py", "image"]
)
def test_1x_files_are_deleted(path):
    # compare exact names: on case-insensitive filesystems (Windows, macOS) the new geneopt/
    # folder makes (ROOT / "GeneOpt").exists() true
    assert path not in {entry.name for entry in ROOT.iterdir()}


def test_no_shims_in_package():
    for path in (ROOT / "geneopt").rglob("*.py"):
        assert not re.search(r"DeprecationWarning|_compat|_LEGACY", path.read_text()), path


def test_readme_has_python_examples():
    assert len(PYTHON_BLOCKS) >= 3


@pytest.mark.parametrize("block", PYTHON_BLOCKS)
def test_readme_code_is_an_example_file_or_runs(block, tmp_path, monkeypatch):
    if block.strip() in EXAMPLES:
        return
    monkeypatch.chdir(tmp_path)
    exec(compile(block, "README.md", "exec"), {"__name__": "readme"})


@pytest.mark.parametrize(
    "stale",
    [
        "Logger",
        "GeneticAlgorithmComparisonType",
        "GeneticAlgorithmCacheType",
        "CachePath",
        "RunNServer",
        "number_of_population",
        "r_cross",
        "debug=True",
        "blob/master/image",
    ],
)
def test_readme_has_no_1x_leftovers(stale):
    assert not re.search(rf"\b{re.escape(stale)}\b", README)  # word match: getLogger is fine


def test_readme_images_exist():
    for name in re.findall(r"docs/images/([\w.-]+\.png)", README):
        assert (ROOT / "docs" / "images" / name).exists(), name


def test_changelog_links_removal_spec():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "## [2.0.0]" in changelog and "specs/011-removed-in-2.0.md" in changelog
