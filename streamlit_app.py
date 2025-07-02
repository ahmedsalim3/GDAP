"""
Main entry point for Streamlit deployment
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

app_dir = PROJECT_ROOT / "app"
sys.path.insert(0, str(app_dir))
os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)

from app import run

if __name__ == "__main__":
    run()
