"""Desktop launch entry point — no console window."""
import sys
from pathlib import Path

# Add backend src to path so the package is importable without install
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from insar_viewer.launcher import run

run(dev_mode=False)
