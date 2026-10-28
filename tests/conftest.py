import os, sys, tempfile
from pathlib import Path
os.environ["CV_DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
