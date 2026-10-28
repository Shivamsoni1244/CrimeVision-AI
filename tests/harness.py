"""Test harness. Stand-ins ONLY for modules that are not part of the supplied sources (storage, validation,
llm.grounding, reports.pdf) and for torch-backed model modules (no torch / no weights / no network here).
Everything else - app.py, fusion, tracking, video, groq_client validation, imaging, detector post-processing -
is the real code."""
import os, re, sys, tempfile, types, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CV_DATA_DIR"] = tempfile.mkdtemp(prefix="cvdata_")
os.environ.pop("GROQ_API_KEY", None)


class ValidationError(Exception): pass


def _mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m


EXT = {"image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}, "video": {".mp4", ".avi", ".mov", ".mkv"},
       "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac"}}
_EV, _AUDIT = {}, {}


def _ingest(cid, kind, f):
    ext = Path(f.filename or "").suffix.lower()
    if ext not in EXT[kind]: raise ValidationError(f"Unsupported {kind} type '{ext}'")
    data = f.read()
    if not data: raise ValidationError("Empty file")
    from config import DATA
    eid = "ev_" + uuid.uuid4().hex[:10]
    d = DATA / cid / "originals"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{eid}{ext}"; p.write_bytes(data)
    ev = {"id": eid, "kind": kind, "filename": f.filename, "path": str(p), "sha256": "x"}
    _EV.setdefault(cid, []).append(ev); return ev


storage = _mod("storage", init_db=lambda: None, ingest=_ingest, list_evidence=lambda cid: list(_EV.get(cid, [])),
               verify=lambda cid: {"case": cid, "ok": True},
               audit=lambda cid, a, d="": _AUDIT.setdefault(cid, []).append((a, d)),
               get_audit=lambda cid: _AUDIT.get(cid, []))
validation = _mod("validation", ValidationError=ValidationError)


def _extract(text, ids):
    found = re.findall(r"\b(?:evt|ev)_[A-Za-z0-9]+", text)
    return {"valid": [f for f in found if f in ids], "unknown": [f for f in found if f not in ids]}


import llm  # real package  # noqa: E402
_mod("llm.grounding", extract=_extract); llm.grounding = sys.modules["llm.grounding"]
_mod("reports"); _mod("reports.pdf", generate=lambda c, out: Path(out).write_bytes(b"%PDF-1.4 stub"))

import models  # noqa: E402  (real, empty package)

try:
    import torch  # noqa: F401  (real torch if installed)
except ImportError:  # no torch here: minimal stand-in so models/firearm.py imports; it only needs tensor/no_grad
    class _NoGrad:
        def __enter__(s): return s
        def __exit__(s, *a): return False
    _mod("torch", tensor=lambda x: x, no_grad=lambda: _NoGrad())


class FakeDetector:
    next_dets = []; fail = False; seen_size = None
    @classmethod
    def detect(cls, img, threshold=None, relevant_only=False):
        cls.seen_size = img.size
        if cls.fail: raise RuntimeError("weights missing")
        return [dict(d) for d in cls.next_dets]
    @staticmethod
    def model_info(): return {"model": "fake-rtdetr", "threshold": 0.5, "limitation": "test"}


class FakeScene:
    SCENES = ["a robbery in progress", "a normal peaceful scene"]
    fail_caption = False
    @classmethod
    def classify_scene(cls, img, top_k=3):
        return [{"label": "a normal peaceful scene", "score": 0.7}, {"label": "a robbery in progress", "score": 0.3}][:top_k]
    @classmethod
    def caption(cls, img):
        if cls.fail_caption: raise RuntimeError("blip failed")
        return "a person standing in a room"


det_mod = _mod("models.detector", detect=FakeDetector.detect, model_info=FakeDetector.model_info)
scene_mod = _mod("models.scene", SCENES=FakeScene.SCENES, classify_scene=FakeScene.classify_scene,
                 caption=FakeScene.caption)
models.detector, models.scene = det_mod, scene_mod


def det(label="person", conf=0.9, box=(0.1, 0.2, 0.4, 0.8), relevant=True):
    x1, y1, x2, y2 = box
    return {"label": label, "score": conf, "confidence": conf, "relevant": relevant,
            "weapon": label in ("knife", "baseball bat", "scissors"),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": x2 - x1, "h": y2 - y1}
