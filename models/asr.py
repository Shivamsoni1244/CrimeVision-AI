"""Whisper transcription with timestamped segments.

The ORIGINAL-language transcript is stored (task="transcribe"). An English translation is only added when
WHISPER_TRANSLATE=1, as a separate field, so the original wording is never replaced. Transcripts are
unverified machine output, not witness statements of fact.
"""
import shutil
import threading

from config import WHISPER_MODEL, WHISPER_TRANSLATE

_lock = threading.Lock()
_model = None


class AudioError(Exception):
    """Failure with a message that is safe to show to the user."""


def _load():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    import whisper
                except ImportError:
                    raise AudioError("The 'openai-whisper' package is not installed on the server.")
                _model = whisper.load_model(WHISPER_MODEL)
    return _model


def _segments(r):
    return [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in r.get("segments", []) if s["text"].strip()]


def transcribe(path: str):
    if not shutil.which("ffmpeg"):
        raise AudioError("ffmpeg was not found on PATH; Whisper needs it to read audio files.")
    model = _load()
    try:
        r = model.transcribe(path, task="transcribe")
    except Exception as e:
        import logging
        logging.exception("whisper failed")
        raise AudioError("The audio could not be transcribed (unsupported or corrupt file). See the Flask log.") from e
    out = {"language": r.get("language"), "text": r["text"].strip(), "segments": _segments(r),
           "model": WHISPER_MODEL, "basis": "transcribed_speech_unverified"}
    if not out["segments"]:
        out["note"] = "No speech was detected."
    if WHISPER_TRANSLATE and r.get("language") not in (None, "en"):
        t = model.transcribe(path, task="translate")
        out["translation_en"] = {"text": t["text"].strip(), "segments": _segments(t)}
    return out
