"""
Root-level pytest configuration.

Ensures that `examiner_env` is importable even when the package has not been
installed via `pip install -e ./examiner_env`.  The parent directory of
`examiner_env/` (i.e. the repo root) is prepended to sys.path so that
`import examiner_env` resolves to the live source tree.
"""

import sys
from pathlib import Path

# Repo root  →  contains the `examiner_env/` source directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
