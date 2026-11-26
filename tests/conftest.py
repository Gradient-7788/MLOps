import os
import sys
import tempfile
from pathlib import Path

# Keep every artifact a test creates (models, runs, logs, datasets) out of the source tree.
os.environ.setdefault("MLOPS_HOME", tempfile.mkdtemp(prefix="mlops_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
