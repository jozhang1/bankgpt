"""Make tests import the submission the way the harness does.

The harness puts ``submission/`` on ``sys.path`` and imports ``solution`` as a
top-level module, so every internal import is top-level (``facts.x``,
``retrieval.y``).  Importing through the package name (``submission.solution``)
would load each module twice under two names, which breaks patching and
isinstance checks.  Tests therefore import top-level names, and this file
guarantees ``submission/`` and the harness's ``contract`` module are importable.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for path in (_HERE, _HERE.parent / 'harness'):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
