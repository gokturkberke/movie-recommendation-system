import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_baselines import *  # noqa: F403,E402
from scripts.evaluate_baselines import main  # noqa: E402


if __name__ == "__main__":
    main()
