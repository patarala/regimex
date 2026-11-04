"""Bootstrap the standard-library installer before service dependencies exist."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    # Import the standalone installer without importing regimex's installed metadata.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/regimex"))
    runpy.run_module("local_installation", run_name="__main__")
