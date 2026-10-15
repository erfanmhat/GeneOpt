import subprocess
import sys

import geneopt


def test_version():
    assert geneopt.__version__ == "2.0.0"


def test_import_has_no_side_effects(tmp_path):
    code = (
        "import sys, geneopt\n"
        "heavy = {'matplotlib', 'pandas', 'seaborn', 'catboost', 'sklearn'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    env = {"PYTHONPATH": str(__import__("pathlib").Path(geneopt.__file__).parent.parent)}
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    assert list(tmp_path.iterdir()) == []
