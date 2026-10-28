"""Compatibility adapter: the overlay payload shape, produced by the single RT-DETR detector.

Previously this module loaded a second model (YOLOS), so the boxes drawn in the UI could differ from the
detections given to the LLM. It now reformats the same detections, so both always agree.
"""
from PIL import Image

from config import DETECTOR_MODEL, DETECTOR_THRESHOLD
from models import detector


def boxes_payload(image: Image.Image, dets: list, threshold: float | None = None) -> dict:
    w, h = image.size
    return {"model": DETECTOR_MODEL, "threshold": DETECTOR_THRESHOLD if threshold is None else threshold,
            "width": w, "height": h,
            "boxes": [{"label": d["label"], "confidence": d["confidence"], "box": d["box"]} for d in dets]}


def detect_boxes(image: Image.Image, threshold: float | None = None) -> dict:
    return boxes_payload(image, detector.detect(image, threshold), threshold)
