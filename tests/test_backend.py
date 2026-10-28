import io, json, math, shutil, sys, unittest
from pathlib import Path
from unittest import mock

import tests.harness as H  # noqa: F401  (sets env + stand-in modules first)
import cv2, numpy as np
from PIL import Image

import app as appmod
from llm import groq_client as G
from pipeline import fusion, tracking, video as V
from models import imaging


def jpg(size=(200, 100), color=(120, 40, 40), exif_orientation=None):
    im = Image.new("RGB", size, color); b = io.BytesIO()
    if exif_orientation:
        ex = Image.Exif(); ex[0x0112] = exif_orientation; im.save(b, "JPEG", exif=ex)
    else: im.save(b, "JPEG")
    return b.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.c = appmod.app.test_client()
        H.FakeDetector.next_dets = [H.det()]; H.FakeDetector.fail = False; H.FakeScene.fail_caption = False
        G.GROQ_API_KEY = ""
        G._client = None
    def up(self, path, data, name, case="t1"):
        return self.c.post(path, data={"case_id": case, "file": (io.BytesIO(data), name)},
                           content_type="multipart/form-data")


class ImageTests(Base):
    def test_valid_image_structured_response_without_key(self):
        r = self.up("/api/analyze/image", jpg(), "a.jpg")
        self.assertEqual(r.status_code, 200); j = r.get_json()
        self.assertEqual(j["detections"][0]["label"], "person")
        self.assertEqual(j["object_boxes"]["boxes"][0]["box"], {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.8})
        self.assertEqual(j["scene"][0]["label"], "a normal peaceful scene")
        self.assertTrue(j["caption"]); self.assertTrue(j["image_url"].endswith("/file"))
        self.assertEqual(j["assessment"]["code"], "missing_api_key")        # LLM unavailable != analysis failure
        self.assertNotIn("GROQ", json.dumps(j).replace("GROQ_API_KEY is not set", ""))

    def test_exif_rotated_image_is_analysed_in_displayed_orientation(self):
        r = self.up("/api/analyze/image", jpg((200, 100), exif_orientation=6), "p.jpg")
        j = r.get_json()
        self.assertEqual(H.FakeDetector.seen_size, (100, 200))              # detector saw the transposed image
        self.assertEqual((j["width"], j["height"]), (100, 200))
        self.assertEqual(j["object_boxes"]["width"], 100)
        orig = Path(appmod.storage.list_evidence("t1")[-1]["path"]).read_bytes()
        self.assertEqual(Image.open(io.BytesIO(orig)).size, (200, 100))     # stored original untouched

    def test_missing_file_bad_case_bad_type(self):
        self.assertEqual(self.c.post("/api/analyze/image", data={"case_id": "t1"}).status_code, 400)
        self.assertEqual(self.up("/api/analyze/image", jpg(), "a.jpg", case="bad/../id").status_code, 404 if False else 400)
        r = self.up("/api/analyze/image", b"hello", "a.txt"); self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())
        self.assertEqual(self.up("/api/evidence/zip", b"x", "a.zip").status_code, 400)

    def test_corrupt_image_is_stored_but_reported(self):
        r = self.up("/api/analyze/image", b"not an image at all", "a.jpg")
        self.assertEqual(r.status_code, 400); self.assertIn("could not be decoded", r.get_json()["error"])

    def test_model_failure_keeps_evidence_and_reports_errors(self):
        H.FakeDetector.fail = True; H.FakeScene.fail_caption = True
        j = self.up("/api/analyze/image", jpg(), "a.jpg").get_json()
        self.assertEqual(set(j["errors"]), {"detector", "caption"})
        self.assertEqual(j["detections"], []); self.assertIsNone(j["object_boxes"])
        self.assertIn("scene", j and j)  # scene still present
        self.assertTrue(any(e["id"] == j["evidence_id"] for e in appmod.storage.list_evidence("t1")))

    def test_all_steps_failing_gives_insufficient_without_llm(self):
        H.FakeDetector.fail = True; H.FakeScene.fail_caption = True
        with mock.patch.object(H.scene_mod, "classify_scene", side_effect=RuntimeError("x")):
            j = self.up("/api/analyze/image", jpg(), "a.jpg").get_json()
        self.assertEqual(j["assessment"]["category"], "Insufficient evidence")

    def test_reanalyze_does_not_duplicate_records(self):
        j = self.up("/api/analyze/image", jpg(), "a.jpg", case="t2").get_json()
        for _ in range(2):
            self.assertEqual(self.c.post(f"/api/reanalyze/t2/{j['evidence_id']}").status_code, 200)
        self.assertEqual(len(appmod.load("t2")["images"]), 1)
        self.assertEqual(len(appmod.storage.list_evidence("t2")), 1)
        self.assertEqual(self.c.post("/api/reanalyze/t2/nope").status_code, 404)

    def test_timeline_has_untimed_image_and_no_duplicate_caption_in_context(self):
        self.up("/api/analyze/image", jpg(), "a.jpg", case="t3")
        tl = self.c.get("/api/timeline/t3").get_json()
        self.assertIsNone(tl["timeline"][0]["t"])
        ctx = fusion.build_context(appmod.load("t3"))
        self.assertEqual(ctx.count("a person standing in a room"), 1)


class DetectorMath(unittest.TestCase):
    def test_real_detector_postprocessing_normalises_and_clamps(self):
        fake_torch = types_mod("torch", tensor=lambda x: x, no_grad=lambda: _Ctx())
        import importlib
        sys.modules.pop("models.detector", None)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            d = importlib.import_module("models.detector")
            class B:
                def __init__(s, v): s.v = v
                def tolist(s): return s.v
            class Model:
                class config: id2label = {1: "person", 2: "chair"}
                def __call__(s, **k): return None
            class Proc:
                def __call__(s, images, return_tensors): return {}
                def post_process_object_detection(s, out, target_sizes, threshold):
                    assert target_sizes == [[200, 100]]
                    return [{"scores": [0.91, 0.6], "labels": [1, 2],
                             "boxes": [B([-5, 20, 50, 250]), B([10, 10, 20, 20])]}]
            d._det = (Model(), Proc())
            out = d.detect(Image.new("RGB", (100, 200)))
        self.assertEqual(out[0]["box"], {"x1": 0.0, "y1": 0.1, "x2": 0.5, "y2": 1.0})   # clamped, normalised
        self.assertTrue(out[0]["relevant"]); self.assertFalse(out[1]["relevant"])
        self.assertEqual(out[0]["score"], out[0]["confidence"])
        sys.modules["models.detector"] = H.det_mod; H.models.detector = H.det_mod


class _Ctx:
    def __enter__(s): return s
    def __exit__(s, *a): return False


def types_mod(name, **kw):
    import types; m = types.ModuleType(name); m.__dict__.update(kw); return m


class ClassifyTests(Base):
    def seed(self, cid="c1"):
        self.up("/api/analyze/image", jpg(), "a.jpg", case=cid)
        return appmod.load(cid)["images"][0]["evidence_id"]

    def test_empty_case_is_insufficient_without_llm_call(self):
        self.c.post("/api/cases", json={"case_id": "empty1"})
        with mock.patch.object(G, "_chat") as ch:
            j = self.c.post("/api/classify/empty1").get_json()
        ch.assert_not_called(); self.assertEqual(j["category"], "Insufficient evidence")
        self.assertEqual(j["review_status"], "requires_human_review"); self.assertIn("not a finding", j["disclaimer"])

    def test_missing_api_key_returns_503_json(self):
        self.seed("c2"); r = self.c.post("/api/classify/c2")
        self.assertEqual(r.status_code, 503); self.assertIn("GROQ_API_KEY", r.get_json()["error"])

    def test_crime_label_without_valid_citation_is_downgraded(self):
        eid = self.seed("c3"); G.GROQ_API_KEY = "k"
        reply = json.dumps({"category": "Robbery", "confidence": 0.9, "reasoning": "r",
                            "supporting_observations": [{"text": "A masked man", "citations": ["evt_deadbeef"]}]})
        with mock.patch.object(G, "_chat", return_value=reply):
            j = self.c.post("/api/classify/c3").get_json()
        self.assertEqual(j["category"], "Insufficient evidence"); self.assertLessEqual(j["confidence"], 0.2)
        self.assertTrue(j["validation_notes"]); self.assertEqual(j["supporting_observations"], [])

    def test_supported_category_kept_with_real_citation(self):
        eid = self.seed("c4"); G.GROQ_API_KEY = "k"
        reply = json.dumps({"category": "assault", "confidence": 1.7, "reasoning": "r",
                            "supporting_observations": [{"text": "person detected", "citations": [eid]}],
                            "uncertainties": ["single frame"], "alternative_explanations": "innocent"})
        with mock.patch.object(G, "_chat", return_value=reply):
            j = self.c.post("/api/classify/c4").get_json()
        self.assertEqual(j["category"], "Assault"); self.assertEqual(j["confidence"], 1.0)
        self.assertEqual(j["alternative_explanations"], ["innocent"]); self.assertEqual(j["key_findings"], ["person detected"])
        self.assertEqual(appmod.load("c4")["analysis"]["category"], "Assault")

    def test_unknown_category_and_malformed_json(self):
        self.seed("c5"); G.GROQ_API_KEY = "k"
        with mock.patch.object(G, "_chat", return_value=json.dumps({"category": "Murder"})):
            self.assertEqual(self.c.post("/api/classify/c5").get_json()["category"], "Insufficient evidence")
        with mock.patch.object(G, "_chat", return_value="sorry, no json") as ch:
            r = self.c.post("/api/classify/c5")
        self.assertEqual(r.status_code, 502); self.assertEqual(ch.call_count, 2)       # one retry

    def test_provider_errors_are_translated(self):
        self.seed("c6"); G.GROQ_API_KEY = "k"
        class RateLimitError(Exception): pass
        class APITimeoutError(Exception): pass
        for exc, code, status in [(RateLimitError("sk-SECRET"), 429, 429), (APITimeoutError("t"), 504, 504)]:
            fake = mock.Mock(); fake.chat.completions.create.side_effect = exc; G._client = fake
            r = self.c.post("/api/classify/c6")
            self.assertEqual(r.status_code, status); self.assertNotIn("SECRET", r.get_data(as_text=True))

    def test_chat_validation_and_role_filter(self):
        self.seed("c7"); G.GROQ_API_KEY = "k"
        self.assertEqual(self.c.post("/api/chat/c7", json={"question": "  "}).status_code, 400)
        self.assertEqual(self.c.post("/api/chat/c7", json={"question": "x" * 2001}).status_code, 400)
        with mock.patch.object(G, "_chat", return_value="Insufficient evidence.") as ch:
            j = self.c.post("/api/chat/c7", json={"question": "what happened?"}).get_json()
        self.assertEqual(j["answer"], "Insufficient evidence."); self.assertEqual(len(appmod.load("c7")["chat"]), 2)
    def test_history_filter(self):
        G.GROQ_API_KEY = "k"
        with mock.patch.object(G, "_chat", return_value="ok") as ch:
            G.chat("ctx", [{"role": "system", "content": "evil"}, {"role": "user", "content": "hi"}], "q")
        roles = [m["role"] for m in ch.call_args[0][0]]
        self.assertEqual(roles, ["system", "user", "user"])


class VideoTests(Base):
    def make_video(self, path, frames=30, fps=10, size=(160, 120), moving=True):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for i in range(frames):
            f = np.zeros((size[1], size[0], 3), np.uint8)
            x = 10 + (i * 4 if moving else 0); cv2.rectangle(f, (x, 40), (x + 30, 100), (255, 255, 255), -1)
            w.write(f)
        w.release()
    def detect_for(self, img):
        a = np.array(img)[:, :, 0]; xs = np.where(a.max(axis=0) > 200)[0]
        if not len(xs): return []
        w, h = img.size
        return [H.det("person", 0.9, (xs.min() / w, 40 / h, (xs.max() + 1) / w, 100 / h))]

    def test_timestamps_frames_tracks_and_limits(self):
        p = Path(H.tempfile.mkdtemp()) / "v.avi"; self.make_video(p, frames=30, fps=10)
        fd = Path(H.tempfile.mkdtemp()) / "fr"
        r = V.analyze_video_full(p, every_sec=1.0, frames_dir=fd, detect_fn=self.detect_for,
                                 scene_fn=H.FakeScene.classify_scene, caption_fn=H.FakeScene.caption)
        self.assertEqual([f["t"] for f in r["frames"]], [0.0, 1.0, 2.0])             # 10 fps, every 10th frame
        self.assertEqual([f["frame_index"] for f in r["frames"]], [0, 10, 20])
        for f in r["frames"]: self.assertTrue((fd / f["frame_ref"]).is_file())
        self.assertEqual(len(r["tracks"]), 1); self.assertEqual(r["tracks"][0]["observations"], 3)
        for e in r["events"]: self.assertIn("frame_ref", e); self.assertIsNotNone(e["t"])
        self.assertTrue(any(e["type"] == "people" for e in r["events"]))
        self.assertIn("no actions were recognised", r["summary"]["text"])
        self.assertTrue(any("action-recognition" in l for l in r["limitations"]))
        self.assertFalse(r["summary"]["truncated"])
        r2 = V.analyze_video_full(p, every_sec=0.5, max_samples=2, detect_fn=self.detect_for)
        self.assertTrue(r2["summary"]["truncated"]); self.assertEqual(r2["summary"]["samples"], 2)
        self.assertIn("NOT analysed", r2["summary"]["text"])

    def test_movement_event_is_measurement_not_action(self):
        p = Path(H.tempfile.mkdtemp()) / "m.avi"; self.make_video(p, frames=21, fps=10)
        def jump(img):
            a = np.array(img)[:, :, 0]; xs = np.where(a.max(axis=0) > 200)[0]; w, h = img.size
            return [H.det("person", 0.9, (xs.min() / w, 0.3, (xs.max() + 1) / w, 0.8))]
        r = V.analyze_video_full(p, every_sec=1.0, detect_fn=jump)
        mv = [e for e in r["events"] if e["type"] == "movement"]
        self.assertGreaterEqual(len(mv), 1)                                          # 0.25 jump per sample is flagged
        self.assertTrue(all(e["basis"] == "measurement" for e in mv))
        self.assertFalse(any(w in e["text"].lower() for e in r["events"] for w in ("assault", "robbery", "attack", "fight")))

    def test_weapon_class_event_uses_non_accusatory_type(self):
        p = Path(H.tempfile.mkdtemp()) / "k.avi"; self.make_video(p, frames=12, fps=10)
        r = V.analyze_video_full(p, every_sec=1.0, detect_fn=lambda i: [H.det("knife", 0.6)])
        ev = [e for e in r["events"] if e["type"] == "object_of_interest"]
        self.assertEqual(len(ev), 1); self.assertIn("prediction only", ev[0]["text"]); self.assertNotIn("weapon detected", ev[0]["text"].lower())

    def test_corrupt_empty_and_bad_params(self):
        d = Path(H.tempfile.mkdtemp())
        (d / "bad.avi").write_bytes(b"garbage" * 100); (d / "empty.avi").write_bytes(b"")
        for n in ("bad.avi", "empty.avi", "missing.avi"):
            with self.assertRaises(V.VideoError): V.analyze_video_full(d / n, detect_fn=lambda i: [])
        with self.assertRaises(V.VideoError): V.analyze_video_full(d / "bad.avi", every_sec=0, detect_fn=lambda i: [])

    def test_route_stores_video_and_serves_frame(self):
        p = Path(H.tempfile.mkdtemp()) / "v.avi"; self.make_video(p, frames=25, fps=10)
        H.FakeDetector.next_dets = [H.det()]
        r = self.up("/api/analyze/video", p.read_bytes(), "v.avi", case="vid1"); self.assertEqual(r.status_code, 200)
        j = r.get_json(); self.assertEqual(j["summary"]["samples"], 3); self.assertEqual(j["events"], len(j["event_list"]))
        eid = j["evidence_id"]
        fr = self.c.get(f"/api/cases/vid1/evidence/{eid}/frames/{j['frames'][1]['frame_ref']}")
        self.assertEqual((fr.status_code, fr.mimetype), (200, "image/jpeg"))
        self.assertEqual(self.c.get(f"/api/cases/vid1/evidence/{eid}/frames/..%2Fx.jpg").status_code, 400 if False else 404)
        self.assertEqual(self.c.get(f"/api/cases/vid1/evidence/{eid}/frames/f_999.jpg").status_code, 404)
        orig = self.c.get(f"/api/cases/vid1/evidence/{eid}/file")                       # original video is served
        self.assertEqual(orig.status_code, 200); self.assertEqual(orig.data, p.read_bytes())
        self.assertEqual(self.c.get(f"/api/cases/vid1/evidence/{eid}/file", headers={"Range": "bytes=0-9"}).status_code, 206)
        self.c.post(f"/api/reanalyze/vid1/{eid}")
        case = appmod.load("vid1"); self.assertEqual(len(case["videos"]), 1)
        self.assertEqual(len({e["event_id"] for e in fusion.build_timeline(case)}), len(fusion.build_timeline(case)))
        self.assertIn("No action recognition was run", fusion.build_context(case))
        r = self.up("/api/analyze/video", b"garbage" * 50, "bad.avi", case="vid1")
        self.assertEqual(r.status_code, 422); self.assertIn("error", r.get_json())


class AudioAndMisc(Base):
    def test_audio_failure_is_clean_json(self):
        r = self.up("/api/analyze/audio", b"RIFFxxxx", "a.wav", case="au1")
        self.assertEqual(r.status_code, 422); self.assertIn("error", r.get_json())
        self.assertIn("evidence_id", r.get_json())

    def test_audio_result_stored_with_segments(self):
        from models import asr
        fake = {"language": "hi", "text": "namaste", "segments": [{"start": 0.0, "end": 1.5, "text": "namaste"}],
                "model": "base", "basis": "transcribed_speech_unverified"}
        with mock.patch.object(asr, "transcribe", return_value=fake):
            j = self.up("/api/analyze/audio", b"RIFFxxxx", "a.wav", case="au2").get_json()
        self.assertEqual(j["segments"][0]["end"], 1.5); self.assertEqual(j["language"], "hi")
        tl = [e for e in self.c.get("/api/timeline/au2").get_json()["timeline"] if e["source"] == "audio"]
        self.assertEqual(tl[0]["basis"], "transcribed_speech_unverified")

    def test_ids_and_listing(self):
        for p in ("/api/integrity/..", "/api/audit/a b"): self.assertIn(self.c.get(p).status_code, (400, 404))
        self.assertEqual(self.c.get("/api/integrity/bad.id").status_code, 400)
        (appmod.DATA / "stray").mkdir(exist_ok=True)
        self.assertNotIn("stray", self.c.get("/api/cases").get_json())
        self.assertEqual(self.c.post("/api/cases", json={"case_id": "x y"}).status_code, 400)
        self.c.post("/api/cases", json={"case_id": "dup"}); self.assertEqual(self.c.post("/api/cases", json={"case_id": "dup"}).status_code, 409)

    def test_non_json_errors_are_json(self):
        r = self.c.post("/api/analyze/image", data={"case_id": "t1"}); self.assertEqual(r.mimetype, "application/json")
        self.assertEqual(self.c.get("/api/nope").mimetype, "application/json")

    def test_report_downloads(self):
        self.up("/api/analyze/image", jpg(), "a.jpg", case="rp1")
        r = self.c.get("/api/report/rp1"); self.assertEqual(r.status_code, 200); self.assertTrue(r.data.startswith(b"%PDF"))


class TrackerTests(unittest.TestCase):
    def test_same_class_association_and_gap(self):
        tr = tracking.IoUTracker(); b = lambda x: {"x1": x, "y1": .2, "x2": x + .2, "y2": .8}
        d = lambda x, l="person": {"label": l, "confidence": .9, "box": b(x)}
        a = tr.update(0, 0, [d(.1)]); self.assertTrue(a[0][2])
        b2 = tr.update(1, 10, [d(.15)]); self.assertFalse(b2[0][2]); self.assertEqual(a[0][1].id, b2[0][1].id)
        c = tr.update(2, 20, [d(.15, "car")]); self.assertTrue(c[0][2])             # other class -> new track
        for i in range(3): tr.update(3 + i, 30 + i, [])
        self.assertEqual(len(tr.finished), 2)


if __name__ == "__main__":
    unittest.main()
