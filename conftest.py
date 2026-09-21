import os
import sys
from pathlib import Path

# MUST be set before torch (via stable_baselines3, imported by
# tests/agents/) or xgboost (imported by tests/baselines/) initialize their
# native thread pools -- running both in the same process without this
# segfaults reliably on this machine (confirmed: importing/using
# stable_baselines3 then calling XGBClassifier.fit() in one process crashes
# with SIGSEGV; XGBClassifier(n_jobs=1) alone does NOT fix it, only
# constraining OMP_NUM_THREADS globally does). This is a real conflict
# between two different OpenMP runtimes both trying to spin up thread pools
# in one process, not a bug in this project's code -- see
# tests/baselines/test_baselines.py / tests/agents/test_agents.py, whichever
# runs second in a full `pytest tests/` invocation, for what would otherwise
# crash. Single-threaded is a negligible slowdown at this project's data
# sizes (hundreds of rows).
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure the repo root is importable as `src.<module>` regardless of how
# pytest is invoked (plain `pytest`, `python -m pytest`, from a subdirectory,
# etc.) -- tests/ has no __init__.py packages, so pytest's default import
# mode doesn't guarantee the repo root ends up on sys.path otherwise.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
