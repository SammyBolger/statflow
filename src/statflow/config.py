"""Project-wide constants: paths and API base URL.

Kept intentionally simple — plain module-level constants, no env vars or
settings framework. If a value ever needs to differ between local dev and
CI, we'll switch to pydantic-settings then.
"""

from pathlib import Path

# Project root — the directory that contains pyproject.toml.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local parquet lake (gitignored).
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# MLB Stats API.
MLB_API_BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1  # 1 = MLB (the API is shared across MiLB, college, etc.)
