"""Runs the REAL app.py with the test stand-ins (no torch/weights here) so the React client can be exercised."""
import sys; sys.path.insert(0, "."); import tests.harness as H
from unittest import mock
from models import asr
asr.transcribe = lambda p: {"language": "hi", "text": "namaste", "segments": [{"start": 0.0, "end": 1.5, "text": "namaste"}], "model": "base", "basis": "transcribed_speech_unverified"}
H.FakeDetector.next_dets = [H.det("person", 0.91, (0.1, 0.2, 0.4, 0.8)), H.det("chair", 0.7, (0.5, 0.5, 0.9, 0.9), relevant=False)]
import app
app.app.run(port=5055, threaded=True)
