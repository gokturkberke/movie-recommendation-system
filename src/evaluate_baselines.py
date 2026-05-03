"""Backward-compatible re-export shim.

Evaluation logic now lives in ``src/evaluation_runner.py``. This module forwards
its public names so existing callers doing ``from evaluate_baselines import ...``
keep working. The CLI entry point still lives at ``scripts/evaluate_baselines.py``;
``main`` and ``build_arg_parser`` are re-exported here so ``python -m evaluate_baselines``
continues to work.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_runner import *  # noqa: F401,F403,E402
from scripts.evaluate_baselines import build_arg_parser, main  # noqa: F401,E402


if __name__ == "__main__":
    main()
