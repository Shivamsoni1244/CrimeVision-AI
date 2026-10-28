"""Frame-sampled video analysis -> frame records, tracks, chronological events and a timestamped summary.

What this does: samples frames at a configurable interval (sequential grab(), never the whole video in memory),
runs the detector / CLIP / BLIP on sampled frames, tracks boxes between samples and reports MEASURABLE changes
(objects appearing/leaving, people count changes, fast box movement, scene-score label changes, captions).
What this does NOT do: recognise actions (assault, theft, ...). No temporal action-recognition model is
integrated, so no event here claims that a crime or violent act occurred.
"""
import math
import re
from pathlib import Path

import cv2
from PIL import Image

from config import VIDEO_CAPTION_EVERY, VIDEO_MAX_SAMPLES, VIDEO_SAMPLE_SEC
from pipeline.tracking import IoUTracker

LIMITATIONS = [
    "No action-recognition model is integrated: events are object presence, count and movement measurements, "
    "not recognised actions.",
    "Detections come from a COCO-trained detector with no firearm class.",
    "Tracks associate boxes between sampled frames only; ids are not person identities and may split or swap.",
    "Timestamps are frame_index / reported fps; unreliable if the file reports a wrong frame rate.",
]
MOVE_THRESHOLD = 0.2  # fraction of frame between consecutive samples
FRAME_NAME = re.compile(r"^f_\d{1,9}\.jpg$")


class VideoError(Exception):
    """Unusable video; message is safe to show to the user."""


def _default_fns():
    from models import detector, scene
    return detector.detect, scene.classify_scene, scene.caption


def _auto_firearm():
    """firearm_fn(img, base_dets) -> merged dets, or None when no firearm model is configured."""
    from models import firearm
    if not firearm.configured():
        return None
    return lambda img, base: firearm.merge(base, firearm.detect(img))


def _event(t, idx, etype, text, basis="measurement", **kw):
    e = {"t": t, "source": "video", "type": etype, "text": text, "frame_index": idx,
         "frame_ref": f"f_{idx}.jpg", "basis": basis}
    e.update({k: v for k, v in kw.items() if v is not None})
    return e


def analyze_video_full(path, every_sec=None, caption_every=None, max_samples=None, frames_dir=None,
                       detect_fn=None, scene_fn=None, caption_fn=None, firearm_fn=None):
    every_sec = VIDEO_SAMPLE_SEC if every_sec is None else every_sec
    caption_every = VIDEO_CAPTION_EVERY if caption_every is None else caption_every
    max_samples = VIDEO_MAX_SAMPLES if max_samples is None else max_samples
    if every_sec <= 0 or max_samples <= 0:
        raise VideoError("Invalid sampling parameters.")
    if detect_fn is None and scene_fn is None and caption_fn is None:
        detect_fn, scene_fn, caption_fn = _default_fns()
        firearm_fn = firearm_fn or _auto_firearm()
    firearm_ran, firearm_frames, firearm_err = bool(firearm_fn), 0, None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("The video could not be opened (corrupt or unsupported format).")
    warnings, errors = [], {}
    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = raw_fps if raw_fps and math.isfinite(raw_fps) and raw_fps > 0 else 25.0
        if fps != raw_fps:
            warnings.append("File reports no valid frame rate; timestamps assume 25 fps.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        step = max(int(round(fps * every_sec)), 1)
        if frames_dir:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)

        tracker, frames, events = IoUTracker(), [], []
        prev_people, prev_scene, prev_pos = None, None, {}
        idx, n, fails, truncated, last_t = 0, 0, 0, False, 0.0
        while True:
            if not cap.grab():
                break
            if idx % step:
                idx += 1
                continue
            if n >= max_samples:
                truncated = True
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                idx += 1
                continue
            t = round(idx / fps, 2)
            last_t = t
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames_dir:
                th = frame
                scale = 640 / max(frame.shape[:2])
                if scale < 1:
                    th = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(Path(frames_dir) / f"f_{idx}.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rec = {"frame_index": idx, "t": t, "frame_ref": f"f_{idx}.jpg", "detections": [], "scene": []}

            dets = []
            if detect_fn:
                try:
                    dets = detect_fn(img)
                    rec["detections"] = [{k: d[k] for k in ("label", "confidence", "box", "relevant", "source") if k in d}
                                         for d in dets]
                except Exception as e:  # keep going; record once
                    fails += 1
                    errors.setdefault("detector", f"{type(e).__name__} at t={t}s (see Flask log)")
                    rec["error"] = "detector failed"
                    import logging; logging.exception("video detector failed at t=%s", t)
            if firearm_fn and not firearm_err:
                try:
                    merged = firearm_fn(img, dets)
                    if any(d.get("firearm") for d in merged):
                        firearm_frames += 1
                    dets = merged
                    rec["detections"] = [{k: d[k] for k in ("label", "confidence", "box", "relevant", "source", "overlaps_coco") if k in d}
                                         for d in dets]
                except Exception as e:  # reported once; COCO analysis continues
                    firearm_err = getattr(e, "args", [None])[0] if type(e).__name__ == "FirearmModelError" else f"{type(e).__name__} at t={t}s (see Flask log)"
                    errors["firearm"] = firearm_err
                    import logging; logging.exception("firearm detector failed at t=%s", t)
            if scene_fn:
                try:
                    rec["scene"] = scene_fn(img, 3)
                except Exception as e:
                    errors.setdefault("scene", f"{type(e).__name__} at t={t}s (see Flask log)")
            if fails >= 3 and not rec["detections"] and "detector" in errors and n < 3:
                raise VideoError("The object detector failed on the first frames; video analysis aborted. "
                                 "Check the Flask log and model download.")

            tracked = tracker.update(t, idx, [d for d in dets if d.get("relevant", True)])
            people = sum(1 for d, _, _ in tracked if d["label"] == "person")
            if prev_people is not None and people != prev_people:
                events.append(_event(t, idx, "people", f"{people} person detection(s) in sampled frame "
                                     f"(was {prev_people})"))
            elif prev_people is None and people:
                events.append(_event(t, idx, "people", f"{people} person detection(s) in first sampled frame"))
            prev_people = people
            for d, tr, is_new in tracked:
                tid = f"T{tr.id}"
                if is_new and d["label"] != "person":
                    note = (" (firearm-model class; prediction only, not verified)" if d.get("firearm") else
                            " (class name that can be a weapon; prediction only)" if d.get("weapon") else "")
                    etype = "object_of_interest" if (d.get("weapon") or d.get("firearm")) else "object_appeared"
                    events.append(_event(t, idx, etype, f"{d['label']} {tid} first detected, "
                                         f"confidence {d['confidence']}{note}", "model_output",
                                         confidence=d["confidence"], track_id=tid, box=d["box"]))
                c = ((d["box"]["x1"] + d["box"]["x2"]) / 2, (d["box"]["y1"] + d["box"]["y2"]) / 2)
                if tr.id in prev_pos:
                    moved = math.dist(c, prev_pos[tr.id])
                    if moved >= MOVE_THRESHOLD:
                        events.append(_event(t, idx, "movement", f"{d['label']} {tid} box centre moved "
                                             f"{moved:.2f} of the frame between consecutive samples",
                                             confidence=d["confidence"], track_id=tid, box=d["box"]))
                prev_pos[tr.id] = c
            if rec["scene"]:
                top = rec["scene"][0]
                if top["label"] != prev_scene and top["score"] >= 0.35:
                    events.append(_event(t, idx, "scene_score", f"CLIP top zero-shot prompt is now "
                                         f"'{top['label']}' (relative score {top['score']}; not a probability "
                                         "that this occurred)", "model_output", confidence=top["score"]))
                    prev_scene = top["label"]
            if caption_fn and caption_every > 0 and n % caption_every == 0:
                try:
                    rec["caption"] = caption_fn(img)
                    events.append(_event(t, idx, "scene", rec["caption"], "model_output"))
                except Exception as e:
                    errors.setdefault("caption", f"{type(e).__name__} at t={t}s (see Flask log)")
            frames.append(rec)
            n += 1
            idx += 1
        if n == 0:
            raise VideoError("No decodable frames were found in this video (empty or corrupt file).")
    finally:
        cap.release()

    tracks = tracker.all_tracks()
    for tr in tracks:
        if tr.label != "person" or tr.n_obs >= 2:
            events.append(_event(tr.last_t, tr.last_frame, "object_left" if tr.last_t < last_t else "object_last_seen",
                                 f"{tr.label} T{tr.id} last detected (seen in {tr.n_obs} sample(s), "
                                 f"{tr.first_t}s-{tr.last_t}s)", "measurement", track_id=f"T{tr.id}"))
    events.sort(key=lambda e: (e["t"], e["frame_index"]))
    by_label = {}
    for tr in tracks:
        by_label[tr.label] = by_label.get(tr.label, 0) + 1
    summary = {"samples": n, "sample_interval_s": every_sec, "analysed_until_s": last_t,
               "duration_s": round(total / fps, 2) if total else None, "truncated": truncated,
               "tracks_by_class": by_label,
               "event_counts": {k: sum(1 for e in events if e["type"] == k) for k in {e["type"] for e in events}},
               "text": _summary_text(n, every_sec, last_t, by_label, events, truncated)}
    return {"events": events, "frames": frames, "tracks": [t.to_dict() for t in tracks], "summary": summary,
            "video": {"fps": round(fps, 3), "frame_count": total, "width": width, "height": height},
            "params": {"every_sec": every_sec, "caption_every": caption_every, "max_samples": max_samples},
            "errors": errors, "warnings": warnings, "limitations": LIMITATIONS,
            "firearm": {"ran": firearm_ran and not firearm_err, "frames_with_firearm_class": firearm_frames, "error": firearm_err}}


def _summary_text(n, every, last_t, by_label, events, truncated):
    parts = [f"Analysed {n} sampled frame(s) every {every}s up to t={last_t}s."]
    if truncated:
        parts.append("Analysis stopped at the sample limit; later footage was NOT analysed.")
    if by_label:
        parts.append("Tracked objects: " + ", ".join(f"{v} {k}" for k, v in sorted(by_label.items())) + ".")
    notable = [e for e in events if e["type"] in ("people", "movement", "object_of_interest")][:6]
    for e in notable:
        parts.append(f"[{e['t']}s] {e['text']}.")
    parts.append("These are object/count/movement observations only; no actions were recognised.")
    return " ".join(parts)


def analyze_video(path: str, every_sec: float = 1.0, caption_every: int = 5):
    """Backwards-compatible: events only."""
    return analyze_video_full(path, every_sec, caption_every)["events"]
