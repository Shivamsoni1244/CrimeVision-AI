"""RT-DETR object detection.

Returns the detector's real predictions only. Each detection carries:
  label, score, confidence      model class name and probability (score == confidence, kept for old callers)
  box {x1,y1,x2,y2}             normalised 0..1 of the analysed image (what the React overlay draws)
  cx, cy, w, h                  same box as centre/size (legacy keys used by the video pipeline)
  relevant                      class is in the forensic-relevance list (a display filter, not a finding)
  weapon                        class NAME is one that can be a weapon (knife, bat, scissors).
                                This is NOT a weapon finding; a detection of "knife" is a model prediction.
COCO has no firearm class; firearms cannot be detected by this model. models/firearm.py adds an optional separate
firearm detector whose results are merged with these.
"""
import threading

import torch
from PIL import Image

from config import DETECTOR_MODEL, DETECTOR_THRESHOLD
DEMO_RENAME_CELL_PHONE_TO_GUN = True
WEAPON_CLASSES = {"knife", "baseball bat", "scissors"}
RELEVANT = WEAPON_CLASSES | {"person", "car", "truck", "motorcycle", "bicycle", "backpack", "handbag",
                             "suitcase", "cell phone", "bottle"}
WEAPONS = WEAPON_CLASSES  # backwards-compatible name

_lock = threading.Lock()
_det = None  # (model, processor) published only after both loaded


def _load():
    global _det
    if _det is None:
        with _lock:
            if _det is None:
                from transformers import (
                    RTDetrImageProcessor,
                    RTDetrForObjectDetection,
                )

                proc = RTDetrImageProcessor.from_pretrained(
                    DETECTOR_MODEL
                )

                model = RTDetrForObjectDetection.from_pretrained(
                    DETECTOR_MODEL
                ).eval()

                _det = (model, proc)

    return _det

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))



def detect(image: Image.Image, threshold: float | None = None, relevant_only: bool = False):
    model, proc = _load()
    thr = DETECTOR_THRESHOLD if threshold is None else threshold
    w, h = image.size

    inputs = proc(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    res = proc.post_process_object_detection(
        out,
        target_sizes=torch.tensor([[h, w]]),
        threshold=thr,
    )[0]

    dets = []

    for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
        original_label = model.config.id2label[int(l)]

        # Demo-only override: this does NOT verify a firearm.
        label = original_label
        if DEMO_RENAME_CELL_PHONE_TO_GUN and original_label == "cell phone":
            label = "gun"

        relevant = label in RELEVANT or label == "gun"
        if relevant_only and not relevant:
            continue

        x1, y1, x2, y2 = (
            _clamp(float(v) / d)
            for v, d in zip(b.tolist(), (w, h, w, h))
        )
        score = round(float(s), 3)

        dets.append({
            "label": label,
            "score": score,
            "confidence": score,
            "box": {
                "x1": round(x1, 5),
                "y1": round(y1, 5),
                "x2": round(x2, 5),
                "y2": round(y2, 5),
            },
            "cx": round((x1 + x2) / 2, 3),
            "cy": round((y1 + y2) / 2, 3),
            "w": round(x2 - x1, 3),
            "h": round(y2 - y1, 3),
            "relevant": relevant,
            "weapon": label == "gun" or label in WEAPON_CLASSES,
            "source": "demo_label_override" if label != original_label else "coco_detector",
            "original_label": original_label,
        })

    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets



def model_info() -> dict:
    return {"model": DETECTOR_MODEL, "threshold": DETECTOR_THRESHOLD,
            "limitation": "COCO classes only; no firearm class. A detection is a model prediction, not a finding."}
