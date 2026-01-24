import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Main source tree lives under `scripts/src/`
SCRIPTS_SRC_DIR = SCRIPTS_DIR / "src"
if SCRIPTS_SRC_DIR.exists() and str(SCRIPTS_SRC_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_SRC_DIR))
