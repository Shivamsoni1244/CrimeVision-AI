import json, logging, mimetypes, os, re, threading, uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory, render_template
from werkzeug.exceptions import HTTPException
from config import DATA, MAX_UPLOAD_MB
import storage, validation
from llm import grounding
from pipeline.fusion import build_timeline, build_context, build_image_context

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
storage.init_db()

CID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
EID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FRAME = re.compile(r"^f_\d{1,9}\.jpg$")
DIST = Path(__file__).parent / "frontend" / "dist"          # built React app (npm run build)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
KINDS = {"image", "video", "audio"}
MAX_QUESTION = 2000
_LOCK = threading.RLock()                                    # guards case.json read-modify-write

def _path(cid): return DATA / cid / "case.json"
def _ok(cid): return bool(CID.match(cid))
def _blank(cid): return {"id": cid, "images": [], "video_events": [], "videos": [], "audio": [], "chat": []}

def load(cid):
    if not _ok(cid): return None
    p = _path(cid)
    if not p.exists(): return None
    case = json.loads(p.read_text())
    for k, v in _blank(cid).items():                        # older cases lack newer keys
        case.setdefault(k, v)
    return case

def save(case):
    """Atomic write: a crash mid-write can never leave a truncated case.json."""
    p = _path(case["id"]); tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(case, indent=2)); os.replace(tmp, p)

def update_case(cid, fn):
    """Reload -> mutate -> save under a lock, so slow model work never needs the lock and no update is lost."""
    with _LOCK:
        case = load(cid) or _blank(cid)
        out = fn(case); save(case); return out

def _public(ev):
    """Evidence record without the server filesystem path."""
    return {k: v for k, v in ev.items() if k != "path"}

def _image_view(cid, im):
    return dict(im, image_url=f"/api/cases/{cid}/evidence/{im['evidence_id']}/file")

def _step(fn, *args):
    """Run one analysis step; a failing model must not lose the already-stored evidence."""
    try:
        return fn(*args), None
    except Exception as e:
        logging.exception("analysis step %s failed", getattr(fn, "__name__", "?"))
        return None, f"{type(e).__name__} (see Flask log)"

def _evidence(cid, eid):
    return next((e for e in storage.list_evidence(cid) if e["id"] == eid), None)

# ---- JSON errors for every /api route (so the frontend never receives an HTML error page) ----
@app.errorhandler(HTTPException)
def http_error(e):
    if request.path.startswith("/api/"):
        msg = (f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB." if e.code == 413
               else e.description)
        return jsonify({"error": msg}), e.code
    return e

@app.errorhandler(Exception)
def unhandled(e):
    if app.debug: raise e
    logging.exception("Unhandled error")
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error. See the Flask terminal for details."}), 500
    return "Internal server error", 500

def ingest(kind):
    if kind not in KINDS: return None, None, ({"error": "kind must be image, video or audio"}, 400)
    f = request.files.get("file")
    if not f: return None, None, ({"error": "file required"}, 400)
    cid = request.form.get("case_id") or "case_" + uuid.uuid4().hex[:6]
    if not CID.match(cid): return None, None, ({"error": "invalid case_id"}, 400)
    try: ev = storage.ingest(cid, kind, f)
    except validation.ValidationError as e: return None, None, ({"error": str(e)}, 400)
    case = update_case(cid, lambda c: c)                    # creates case.json if new, otherwise unchanged
    return case, ev, None

def _llm_error(e):
    return {"error": e.message, "code": e.code}, e.status

# ================= analysis units (shared by analyze + re-analyze routes) =================
def run_image(cid, ev):
    """Returns (record, None) or (None, error_response). Evidence file is only read, never modified."""
    from models import detector, scene, boxes, imaging
    from llm import groq_client
    try: img = imaging.load_rgb(ev["path"])                 # EXIF-transposed RGB: matches what browsers display
    except Exception:
        storage.audit(cid, "analyze_image_failed", ev["id"])
        return None, ({"error": "File was stored but could not be decoded as an image.", "evidence_id": ev["id"]}, 400)
    dets, det_err = _step(detector.detect, img)
    dets, fire_info, fire_err = _firearm_step(img, dets)
    scene_cls, scene_err = _step(lambda i: scene.classify_scene(i, len(scene.SCENES)), img)
    cap, cap_err = _step(scene.caption, img)
    rec = {"evidence_id": ev["id"], "file": ev["filename"], "evidence_type": "image",
           "width": img.size[0], "height": img.size[1],
           "detections": dets or [], "firearm_detector": fire_info, "scene": scene_cls or [], "caption": cap or "",
           "object_boxes": boxes.boxes_payload(img, dets) if dets is not None else None,
           "model_info": {"detector": detector.model_info(),
                          "scene": "CLIP zero-shot; relative scores over a fixed prompt list",
                          "caption": "BLIP image captioning"},
           "processed_at": groq_client.stamp()["generated_at"]}
    errors = {k: v for k, v in {"detector": det_err, "firearm": fire_err, "scene": scene_err, "caption": cap_err}.items() if v}
    if errors: rec["errors"] = errors
    rec["assessment"] = _assess(cid, rec)
    return rec, None

def _firearm_step(img, dets):
    """Run the optional separate firearm model and merge its detections with the COCO ones.
    Returns (detections, info, error). Never raises; 'not_configured' is reported, never presented as 'none found'."""
    from models import firearm
    if not firearm.configured():
        return dets, firearm.status(), None
    try:
        found = firearm.detect(img)
    except Exception as e:
        logging.exception("firearm detector failed")
        msg = str(e) if type(e).__name__ == "FirearmModelError" else f"{type(e).__name__} (see Flask log)"
        return dets, {"status": "error", "model": firearm.FIREARM_MODEL, "error": msg}, msg
    info = dict(firearm.status(), status="ok", found=len(found))
    return firearm.merge(dets or [], found), info, None

def _assess(cid, rec):
    """Preliminary per-image assessment. Never fails the analysis: problems become a status object."""
    from llm import groq_client
    if not (rec["detections"] or rec["scene"] or rec["caption"]):
        return groq_client.insufficient("Every image-analysis step failed, so there are no observations to assess.")
    try:
        out = groq_client.classify(build_image_context(cid, rec))
        out["generated_by"] = groq_client.stamp()
        return out
    except groq_client.LLMError as e:
        return {"status": "unavailable", "error": e.message, "code": e.code}

def run_video(cid, ev):
    from pipeline.video import analyze_video_full, VideoError
    from models import firearm
    frames_dir = DATA / cid / "derived" / ev["id"]
    try: res = analyze_video_full(ev["path"], frames_dir=frames_dir)
    except VideoError as e:
        storage.audit(cid, "analyze_video_failed", ev["id"])
        return None, ({"error": str(e), "evidence_id": ev["id"]}, 422)
    ts = [dict(e, evidence_id=ev["id"]) for e in res["events"]]
    rec = {"evidence_id": ev["id"], "file": ev["filename"], "evidence_type": "video", "summary": res["summary"],
           "video": res["video"], "params": res["params"], "firearm": res["firearm"],
           "firearm_detector": firearm.status(), "tracks": res["tracks"], "frames": res["frames"],
           "errors": res["errors"], "warnings": res["warnings"], "limitations": res["limitations"],
           "events": ts}
    return rec, None

def run_audio(cid, ev):
    from models import asr
    try: tr = asr.transcribe(ev["path"])
    except asr.AudioError as e:
        storage.audit(cid, "analyze_audio_failed", ev["id"])
        return None, ({"error": str(e), "evidence_id": ev["id"]}, 422)
    return dict(tr, evidence_id=ev["id"], file=ev["filename"], evidence_type="audio"), None

def store_result(cid, kind, rec):
    """Replace any earlier analysis of the same evidence id (no duplicate records), then save."""
    def fn(c):
        eid = rec["evidence_id"]
        if kind == "image": c["images"] = [i for i in c["images"] if i["evidence_id"] != eid] + [rec]
        elif kind == "audio": c["audio"] = [a for a in c["audio"] if a.get("evidence_id") != eid] + [rec]
        else:
            c["video_events"] = [e for e in c["video_events"] if e.get("evidence_id") != eid] + rec["events"]
            c["videos"] = [v for v in c["videos"] if v["evidence_id"] != eid] + [{k: v for k, v in rec.items() if k != "events"}]
    update_case(cid, fn)
    storage.audit(cid, f"analyze_{kind}", rec["evidence_id"])

RUNNERS = {"image": run_image, "video": run_video, "audio": run_audio}

def _respond(cid, kind, rec):
    if kind == "image": return jsonify(_image_view(cid, rec) | {"case_id": cid})
    if kind == "video":
        view = {k: v for k, v in rec.items() if k != "events"}
        return jsonify(view | {"case_id": cid, "events": len(rec["events"]), "event_list": rec["events"]})
    return jsonify(rec | {"case_id": cid})

# ================= routes =================
@app.get("/api/config")
def client_config(): return jsonify({"max_upload_mb": MAX_UPLOAD_MB})

@app.post("/api/evidence/<kind>")
def add_evidence(kind):
    case, ev, err = ingest(kind)
    if err: return err
    return jsonify({"case_id": case["id"], "evidence": _public(ev)})

@app.get("/api/cases/<cid>/evidence")
def evidence(cid):
    if not _ok(cid): return {"error": "invalid case_id"}, 400
    return jsonify([_public(e) for e in storage.list_evidence(cid)])

@app.get("/api/cases/<cid>/evidence/<eid>/file")
def evidence_file(cid, eid):
    """Serve the ORIGINAL uploaded image or video (read-only, range requests supported) for the viewer."""
    if not _ok(cid): return {"error": "invalid case_id"}, 400
    ev = _evidence(cid, eid)
    allowed = {"image": IMAGE_EXTS, "video": VIDEO_EXTS}.get(ev["kind"]) if ev else None
    if not allowed: return {"error": "image or video evidence not found"}, 404
    p = Path(ev["path"])
    try: p.resolve().relative_to((DATA / cid / "originals").resolve())
    except ValueError: return {"error": "evidence not found"}, 404
    if p.suffix.lower() not in allowed or not p.is_file():
        return {"error": "evidence not found"}, 404
    resp = send_file(p, mimetype=mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                     conditional=True)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp

@app.get("/api/cases/<cid>/evidence/<eid>/frames/<name>")
def video_frame(cid, eid, name):
    """A sampled frame (derived JPEG) that a timeline entry points to. The original video is untouched."""
    if not (_ok(cid) and EID.match(eid) and FRAME.match(name)): return {"error": "invalid reference"}, 400
    ev = _evidence(cid, eid)
    if not ev or ev["kind"] != "video": return {"error": "video evidence not found"}, 404
    p = DATA / cid / "derived" / eid / name
    if not p.is_file(): return {"error": "frame not found"}, 404
    resp = send_file(p, mimetype="image/jpeg", conditional=True)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp

@app.get("/api/integrity/<cid>")
def integrity(cid):
    if not _ok(cid): return {"error": "invalid case_id"}, 400
    return jsonify(storage.verify(cid))

@app.get("/api/audit/<cid>")
def audit_log(cid):
    if not _ok(cid): return {"error": "invalid case_id"}, 400
    return jsonify(storage.get_audit(cid))

@app.get("/")
def index():
    if (DIST / "index.html").exists(): return send_from_directory(DIST, "index.html")
    return render_template("index.html")                      # legacy page until the React app is built

@app.get("/assets/<path:name>")
def dist_assets(name): return send_from_directory(DIST / "assets", name)

@app.get("/api/cases")
def cases(): return jsonify(sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p / "case.json").exists()))

@app.post("/api/cases")
def create_case():
    cid = ((request.get_json(silent=True) or {}).get("case_id") or "").strip() or "case_" + uuid.uuid4().hex[:6]
    if not _ok(cid): return {"error": "Case ID may use letters, digits, - and _ (max 40 characters)."}, 400
    with _LOCK:
        if _path(cid).exists(): return {"error": "A case with this ID already exists."}, 409
        (DATA / cid).mkdir(parents=True, exist_ok=True)
        save(_blank(cid))
    storage.audit(cid, "case_created", cid)
    return jsonify({"case_id": cid}), 201

def _analyze(kind):
    case, ev, err = ingest(kind)
    if err: return err
    rec, err = RUNNERS[kind](case["id"], ev)
    if err: return err
    store_result(case["id"], kind, rec)
    return _respond(case["id"], kind, rec)

@app.post("/api/analyze/image")
def a_image(): return _analyze("image")

@app.post("/api/analyze/video")
def a_video(): return _analyze("video")

@app.post("/api/analyze/audio")
def a_audio(): return _analyze("audio")

@app.post("/api/reanalyze/<cid>/<eid>")
def reanalyze(cid, eid):
    """Re-run analysis on ALREADY stored evidence (e.g. after a model failure) without re-uploading it.
    The new result replaces the earlier analysis of this evidence id; no duplicate evidence record is made."""
    if not _ok(cid) or not EID.match(eid): return {"error": "invalid id"}, 400
    ev = _evidence(cid, eid)
    if not ev: return {"error": "evidence not found"}, 404
    rec, err = RUNNERS[ev["kind"]](cid, ev)
    if err: return err
    store_result(cid, ev["kind"], rec)
    return _respond(cid, ev["kind"], rec)

@app.get("/api/timeline/<cid>")
def timeline(cid):
    c = load(cid)
    if not c: return {"error": "case not found"}, 404
    return jsonify({"timeline": build_timeline(c),
                    "objects": [dict(d, evidence_id=im["evidence_id"]) for im in c["images"] for d in im["detections"]],
                    "images": [_image_view(cid, im) for im in c["images"]],
                    "videos": c["videos"], "audio": c["audio"],
                    "analysis": c.get("analysis")})

@app.post("/api/classify/<cid>")
def classify(cid):
    from llm import groq_client
    c = load(cid)
    if not c: return {"error": "case not found"}, 404
    try: result = groq_client.classify(build_context(c))
    except groq_client.LLMError as e: return _llm_error(e)
    result["generated_by"] = groq_client.stamp()
    update_case(cid, lambda case: case.__setitem__("analysis", result))
    storage.audit(cid, "classify", result["category"])
    return jsonify(result)

@app.post("/api/chat/<cid>")
def chat(cid):
    from llm import groq_client
    c = load(cid)
    if not c: return {"error": "case not found"}, 404
    q = (request.get_json(silent=True) or {}).get("question", "")
    if not isinstance(q, str) or not q.strip(): return {"error": "question required"}, 400
    if len(q) > MAX_QUESTION: return {"error": f"question too long (max {MAX_QUESTION} characters)"}, 400
    q = q.strip()
    try: a = groq_client.chat(build_context(c), c["chat"], q)
    except groq_client.LLMError as e: return _llm_error(e)
    update_case(cid, lambda case: case["chat"].extend(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}]))
    ids = {e["event_id"] for e in build_timeline(c)} | {e["id"] for e in storage.list_evidence(cid)}
    cites = grounding.extract(a, ids)
    storage.audit(cid, "chat", q[:200])
    return jsonify({"answer": a, "citations": cites["valid"], "unverified_citations": cites["unknown"],
                    "basis": "llm_interpretation", "review_status": "requires_human_review"})

@app.get("/api/report/<cid>")
def report(cid):
    from reports.pdf import generate
    c = load(cid)
    if not c: return {"error": "case not found"}, 404
    out = DATA / cid / "report.pdf"; generate(c, str(out))
    storage.audit(cid, "report_generated", out.name)
    return send_file(out, as_attachment=True, download_name=f"forensic_{cid}.pdf")

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=5000)
