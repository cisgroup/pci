"""Check that every executed notebook matches its jupytext-paired ``.py`` source.

Used by CI (and handy locally). For each ``*.ipynb`` under ``examples/`` and
``case_studies/`` the notebook is converted to ``py:percent`` in memory and compared
with the committed ``.py`` file, ignoring the two header lines jupytext manages itself
(``jupytext_version`` and ``formats``). A mismatch means the ``.py`` was edited without
re-executing / re-syncing the notebook (or the other way round):

    python scripts/check_notebook_sync.py
    cd examples && jupytext --to ipynb --execute 01_four_dof_jacobi_ukf.py   # to fix
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["examples", "case_studies"]
IGNORED = ("jupytext_version:", "formats:")


def _normalise(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.strip().startswith(tuple("# " + k for k in IGNORED))]


def main() -> int:
    bad = 0
    for folder in FOLDERS:
        for nb_path in sorted((ROOT / folder).glob("*.ipynb")):
            py_path = nb_path.with_suffix(".py")
            if not py_path.exists():
                print(f"MISSING  {py_path.relative_to(ROOT)}")
                bad += 1
                continue
            from_nb = _normalise(jupytext.writes(jupytext.read(nb_path), fmt="py:percent"))
            from_py = _normalise(py_path.read_text())
            if from_nb == from_py:
                print(f"in sync  {py_path.relative_to(ROOT)}")
            else:
                bad += 1
                print(f"STALE    {py_path.relative_to(ROOT)}")
                for line in list(difflib.unified_diff(from_nb, from_py, "from .ipynb", "from .py", lineterm=""))[:20]:
                    print("    " + line)
    if bad:
        print(f"\n{bad} pair(s) out of sync. Re-execute with `jupytext --to ipynb --execute <file>.py`.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
