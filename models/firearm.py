"""Optional SEPARATE firearm detector, run next to the COCO detector (models/detector.py).

RT-DETR (COCO) has no firearm class. This module loads a transformers object-detection checkpoint named by
FIREARM_MODEL and reports only what that model predicts. Nothing is configured by default: no model has been
verified for this project, so with FIREARM_MODEL empty every result says "not_configured" rather than "no firearm".

Safety rules enforced here:
  * at load time the checkpoint's id2label must contain firearm-like classes (or FIREARM_LABELS must name them);
    otherwise loading fails loudly instead of silently treating every class as a firearm.
  * only detections of those firearm classes are returned (other classes the model may know are dropped).
  * output uses the same schema/normalisation as detector.detect (box x1..y2 in 0..1 of the analysed image).
  * a detection is a model prediction. It does not show a real firearm, who holds it, or any crime.
"""
import re
import threading

import torch
from PIL import Image

from config import FIREARM_LABELS, FIREARM_MODEL, FIREARM_REVISION, FIREARM_THRESHOLD

_FIREARM_RE = re.compile(r"(gun|pistol|handgun|revolver|firearm|rifle|shotgun|weapon)", re.I)
_lock = threading.Lock()
_state = None  # (model, processor, firearm_label_ids)


class FirearmModelError(Exception):
    """Model could not be used; message is safe to show to the user."""


def configured() -> bool:
    return bool(FIREARM_MODEL)


def select_labels(id2label: dict, override=()) -> dict:
    """Pick which of the model's classes count as firearms. Raises if none can be identified."""
    names = {int(i): str(n) for i, n in id2label.items()}
    if override:
        chosen = {i: n for i, n in names.items() if n.lower() in override}
        missing = [o for o in override if o not in {n.lower() for n in names.values()}]
        if missing:
            raise FirearmModelError(f"FIREARM_LABELS not found in the model's classes: {missing}")
    else:
        chosen = {i: n for i, n in names.items() if _FIREARM_RE.search(n)}
    if not chosen:
        raise FirearmModelError("The configured model has no firearm-like class "
                                f"(its classes: {sorted(names.values())[:20]}). Set FIREARM_LABELS or choose another model.")
    return chosen


def _load():
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                from transformers import AutoImageProcessor, AutoModelForObjectDetection
                kw = {"revision": FIREARM_REVISION} if FIREARM_REVISION else {}
                try:
                    proc = AutoImageProcessor.from_pretrained(FIREARM_MODEL, **kw)
                    model = AutoModelForObjectDetection.from_pretrained(FIREARM_MODEL, use_safetensors=True, **kw).eval()
                except Exception as e:
                    raise FirearmModelError(f"Could not load firearm model '{FIREARM_MODEL}' "
                                            f"({type(e).__name__}); it must be a safetensors transformers "
                                            "object-detection checkpoint. See the Flask log.") from e
                _state = (model, proc, select_labels(model.config.id2label, FIREARM_LABELS))
    return _state


def status() -> dict:
    if not configured():
        return {"status": "not_configured", "note": "No firearm detector is configured, so firearms are NOT "
                "being checked for. RT-DETR has no firearm class."}
    return {"status": "configured", "model": FIREARM_MODEL, "threshold": FIREARM_THRESHOLD,
            "classes": sorted(_state[2].values()) if _state else None,
            "note": "Separate firearm model; predictions can be wrong (false positives on toys, tools, phones; misses)."}


def _clamp(v):
    return max(0.0, min(1.0, v))


def detect(image: Image.Image, threshold: float | None = None):
    """Returns firearm-class detections (possibly empty). Raises FirearmModelError if not configured/loadable."""
    if not configured():
        raise FirearmModelError("No firearm model configured (set FIREARM_MODEL).")
    model, proc, ids = _load()
    thr = FIREARM_THRESHOLD if threshold is None else threshold
    w, h = image.size
    inputs = proc(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_object_detection(out, target_sizes=torch.tensor([[h, w]]), threshold=thr)[0]
    dets = []
    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        if int(l) not in ids:
            continue
        x1, y1, x2, y2 = (_clamp(float(v) / d) for v, d in zip(b.tolist(), (w, h, w, h)))
        score = round(float(s), 3)
        dets.append({"label": ids[int(l)], "score": score, "confidence": score,
                     "box": {"x1": round(x1, 5), "y1": round(y1, 5), "x2": round(x2, 5), "y2": round(y2, 5)},
                     "cx": round((x1 + x2) / 2, 3), "cy": round((y1 + y2) / 2, 3),
                     "w": round(x2 - x1, 3), "h": round(y2 - y1, 3),
                     "relevant": True, "weapon": True, "firearm": True, "source": "firearm_model"})
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def box_iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(base: list, firearms: list, iou_dup: float = 0.6, iou_note: float = 0.4) -> list:
    """Combine COCO detections with firearm-model detections.

    * firearm vs firearm overlaps (IoU >= iou_dup): keep the higher score only (NMS).
    * firearm vs a COCO detection of any class: BOTH are kept (a person box legitimately contains a gun box); when
      they overlap strongly the firearm detection records `overlaps_coco` so a disagreement (e.g. COCO says 'cell
      phone' where the firearm model says a gun) is visible instead of silently resolved.
    """
    kept = []
    for f in sorted(firearms, key=lambda d: d["score"], reverse=True):
        if any(box_iou(f["box"], k["box"]) >= iou_dup for k in kept):
            continue
        kept.append(dict(f))
    for f in kept:
        ov = [b["label"] for b in base if b.get("label") != "person" and box_iou(f["box"], b["box"]) >= iou_note]
        if ov:
            f["overlaps_coco"] = sorted(set(ov))
    out = [dict(d, source=d.get("source", "coco_detector")) for d in base] + kept
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
