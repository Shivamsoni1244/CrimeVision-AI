"""Fuse image/video/audio outputs into one timeline with stable event IDs, and build the LLM context.

Evidence types are never merged across files: every record keeps its own evidence_id, and still images are
"untimed" (never given a fake position on a video timeline).
"""
import hashlib

MAX_CONTEXT_LINES = 400


def _eid(e):
    key = f"{e.get('evidence_id')}|{e['source']}|{e['type']}|{e['t']}|{e['text']}"
    return "evt_" + hashlib.sha1(key.encode()).hexdigest()[:8]


def build_timeline(case: dict):
    ev = [dict(e, basis=e.get("basis") or "model_output") for e in case.get("video_events", [])]
    for tr in case.get("audio", []):
        for s in tr.get("segments", []):
            ev.append({"t": s["start"], "t_end": s["end"], "source": "audio", "type": "statement",
                       "text": s["text"], "evidence_id": tr.get("evidence_id"),
                       "basis": "transcribed_speech_unverified"})
    for im in case.get("images", []):  # still images have no timeline position: t=None (never faked as 0)
        if im.get("caption"):
            ev.append({"t": None, "source": "image", "type": "scene", "text": im["caption"],
                       "evidence_id": im.get("evidence_id"), "basis": "model_output"})
    uniq = {}
    for e in ev:
        e["event_id"] = _eid(e); e.setdefault("review_status", "unreviewed")
        uniq.setdefault(e["event_id"], e)
    return sorted(uniq.values(), key=lambda e: (e["t"] is None, e["t"] or 0))


def _image_line(im):
    dets = im.get("detections") or []
    def one(d):
        tag = f"{d['label']}({d.get('score', d.get('confidence'))})"
        if d.get("firearm"):
            tag += "[FIREARM-MODEL prediction" + (f"; overlaps COCO {'/'.join(d['overlaps_coco'])}" if d.get("overlaps_coco") else "") + "]"
        return tag
    objs = ", ".join(one(d) for d in dets) or "none detected"
    fd = im.get("firearm_detector") or {}
    if fd.get("status") == "not_configured":
        fire = " FIREARM CHECK: not run (no firearm model configured), so absence of a firearm is unknown."
    elif fd.get("status") == "error":
        fire = " FIREARM CHECK: FAILED, result unknown."
    elif fd.get("status") == "ok":
        fire = f" FIREARM CHECK: separate model '{fd.get('model')}' ran; {fd.get('found', 0)} firearm-class detection(s); a miss is still possible."
    else:
        fire = ""
    sc = im.get("scene") or []
    scene = "; ".join(f"{s['label']} ({s['score']})" for s in sc[:3]) or "unavailable"
    errs = im.get("errors") or {}
    err = f" PROCESSING ERRORS (that output is missing, not negative): {', '.join(errs)}." if errs else ""
    return (f"[{im.get('evidence_id')} image {im.get('file')}] MODEL OUTPUTS (not verified facts) - "
            f"caption (BLIP, may be wrong): {im.get('caption') or 'unavailable'}; "
            f"detected objects (RT-DETR COCO classes, which has no firearm class{', plus separate firearm model' if fd.get('status') == 'ok' else ''}): {objs}; "
            f"scene prompt scores (CLIP, relative, sum to 1): {scene}.{fire}{err}")


def build_context(case: dict) -> str:
    lines = [f"CASE {case['id']}"]
    for im in case.get("images", []):
        lines.append(_image_line(im))
    for v in case.get("videos", []):
        s = v.get("summary", {})
        lines.append(f"[{v.get('evidence_id')} video {v.get('file')}] sampled every {s.get('sample_interval_s')}s, "
                     f"{s.get('samples')} frames analysed up to {s.get('analysed_until_s')}s"
                     f"{' (TRUNCATED)' if s.get('truncated') else ''}. No action recognition was run."
                     + (f" ERRORS: {', '.join(v['errors'])}." if v.get("errors") else "")
                     + (" FIREARM CHECK: not run (no firearm model configured)." if (v.get("firearm") or {}).get("ran") is False and not (v.get("firearm") or {}).get("error") else "")
                     + (f" FIREARM CHECK ran; firearm-class detections in {v['firearm']['frames_with_firearm_class']} sampled frame(s)." if (v.get("firearm") or {}).get("ran") else ""))
    timeline = [e for e in build_timeline(case) if e["source"] != "image"]  # images already described above
    for e in timeline[:MAX_CONTEXT_LINES]:
        t = f"{e['t']:.2f}s" if e["t"] is not None else "untimed"
        lines.append(f"[{e['event_id']} {t} {e['source']}/{e['type']} from {e.get('evidence_id')} "
                     f"{e.get('basis')}] {e['text']}")
    if len(timeline) > MAX_CONTEXT_LINES:
        lines.append(f"(timeline truncated: {len(timeline) - MAX_CONTEXT_LINES} later events omitted)")
    return "\n".join(lines)


def build_image_context(case_id: str, im: dict) -> str:
    return build_context({"id": case_id, "images": [im]})
