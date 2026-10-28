import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE = Path(__file__).parent
DATA = Path(os.getenv("CV_DATA_DIR", BASE / "data"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
DATA.mkdir(exist_ok=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Object detector (models/detector.py). RT-DETR by default; any transformers object-detection model id works.
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "PekingU/rtdetr_r50vd")
DETECTOR_THRESHOLD = float(os.getenv("DETECTOR_THRESHOLD", "0.5"))

# Video sampling limits (pipeline/video.py)
VIDEO_SAMPLE_SEC = float(os.getenv("VIDEO_SAMPLE_SEC", "1.0"))
VIDEO_MAX_SAMPLES = int(os.getenv("VIDEO_MAX_SAMPLES", "300"))
VIDEO_CAPTION_EVERY = int(os.getenv("VIDEO_CAPTION_EVERY", "5"))
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "30"))
WHISPER_TRANSLATE = os.getenv("WHISPER_TRANSLATE", "0") == "1"

# Optional separate firearm detector (models/firearm.py). Empty = disabled. Run scripts/verify_firearm_model.py
# on a candidate model id (license, labels, weight format) BEFORE setting this.
FIREARM_MODEL = os.getenv("FIREARM_MODEL", "").strip()
FIREARM_THRESHOLD = float(os.getenv("FIREARM_THRESHOLD", "0.6"))
FIREARM_LABELS = [s.strip().lower() for s in os.getenv("FIREARM_LABELS", "").split(",") if s.strip()]
FIREARM_REVISION = os.getenv("FIREARM_REVISION", "").strip() or None   # pin a commit hash once verified
