"""Firearm-detector integration tests. The model is a FAKE (no weights, no network available), so these test the
integration logic - label verification, coordinate normalisation, merging, pipeline wiring, LLM guards - NOT the
accuracy of any real firearm model."""
import io, json, sys, unittest
from unittest import mock

import tests.harness as H  # noqa: F401
from PIL import Image

import app as appmod
from llm import groq_client as G
from models import firearm as F
from pipeline import fusion, video as V
sys.path.insert(0, str(H.ROOT / "scripts"))
import verify_firearm_model as VF


class B:
    def __init__(s, v): s.v = v
    def tolist(s): return s.v


class FakeFirearmModel:
    class config: id2label = {0: "person", 1: "handgun", 2: "knife"}
    out = []                                              # [(score, label_id, [x1,y1,x2,y2] pixels)]
    def __call__(s, **k): return None


class FakeProc:
    seen = None
    def __call__(s, images, return_tensors): FakeProc.seen = images.size; return {}
    def post_process_object_detection(s, out, target_sizes, threshold):
        keep = [o for o in FakeFirearmModel.out if o[0] >= threshold]
        return [{"scores": [o[0] for o in keep], "labels": [o[1] for o in keep], "boxes": [B(o[2]) for o in keep]}]


def install(out, model="fake/firearm"):
    FakeFirearmModel.out = out
    F._state = (FakeFirearmModel(), FakeProc(), F.select_labels(FakeFirearmModel.config.id2label))
    F.FIREARM_MODEL = model


class Base(unittest.TestCase):
    def setUp(self):
        self.c = appmod.app.test_client(); F._state = None; F.FIREARM_MODEL = ""
        H.FakeDetector.next_dets = [H.det("person", 0.9, (0.1, 0.1, 0.6, 0.95)), H.det("cell phone", 0.8, (0.30, 0.40, 0.40, 0.50))]
        H.FakeDetector.fail = False; G.GROQ_API_KEY = ""; G._client = None
    def tearDown(self): F._state = None; F.FIREARM_MODEL = ""
    def up(self, data, name="a.jpg", case="fx", kind="image"):
        return self.c.post(f"/api/analyze/{kind}", data={"case_id": case, "file": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    def img(self, size=(200, 100)):
        b = io.BytesIO(); Image.new("RGB", size, (80, 80, 80)).save(b, "JPEG"); return b.getvalue()


class LabelSelection(unittest.TestCase):
    def test_selects_only_firearm_like_classes(self):
        self.assertEqual(F.select_labels({0: "person", 1: "Handgun", 2: "knife", 3: "rifle"}), {1: "Handgun", 3: "rifle"})
    def test_model_without_firearm_class_is_rejected(self):
        with self.assertRaises(F.FirearmModelError): F.select_labels({0: "person", 1: "car"})
    def test_explicit_override_and_missing_override(self):
        self.assertEqual(F.select_labels({0: "a", 1: "pst"}, ("pst",)), {1: "pst"})
        with self.assertRaises(F.FirearmModelError): F.select_labels({0: "a"}, ("pst",))


class DetectAndMerge(Base):
    def test_not_configured_is_not_reported_as_none_found(self):
        with self.assertRaises(F.FirearmModelError): F.detect(Image.new("RGB", (10, 10)))
        j = self.up(self.img()).get_json()
        self.assertEqual(j["firearm_detector"]["status"], "not_configured")
        self.assertNotIn("firearm", j.get("errors", {}))
        self.assertIn("FIREARM CHECK: not run", fusion.build_context(appmod.load("fx")))

    def test_positive_boxes_normalised_and_aligned(self):
        install([(0.91, 1, [60, 20, 100, 60]), (0.95, 0, [0, 0, 50, 50]), (0.3, 1, [0, 0, 10, 10])])
        out = F.detect(Image.new("RGB", (200, 100)))
        self.assertEqual(FakeProc.seen, (200, 100))
        self.assertEqual(len(out), 1)                                     # person class dropped, low score filtered
        d = out[0]
        self.assertEqual((d["label"], d["score"], d["firearm"], d["source"]), ("handgun", 0.91, True, "firearm_model"))
        self.assertEqual(d["box"], {"x1": 0.3, "y1": 0.2, "x2": 0.5, "y2": 0.6})
        install([(0.9, 1, [-10, 20, 260, 140])]); b = F.detect(Image.new("RGB", (200, 100)))[0]["box"]
        self.assertEqual(b, {"x1": 0.0, "y1": 0.2, "x2": 1.0, "y2": 1.0})  # clamped

    def test_negative_example_returns_empty_but_status_ok(self):
        install([(0.95, 0, [0, 0, 50, 50]), (0.8, 2, [5, 5, 30, 30])])    # only person/knife classes
        self.assertEqual(F.detect(Image.new("RGB", (200, 100))), [])
        j = self.up(self.img()).get_json()
        self.assertEqual((j["firearm_detector"]["status"], j["firearm_detector"]["found"]), ("ok", 0))
        self.assertFalse(any(d.get("firearm") for d in j["detections"]))
        self.assertIn("a miss is still possible", fusion.build_context(appmod.load("fx")))

    def test_merge_keeps_coco_flags_overlap_and_dedupes_firearm_boxes(self):
        base = [H.det("person", .9, (.1, .1, .6, .95)), H.det("cell phone", .8, (.30, .40, .40, .50))]
        f1 = dict(H.det("handgun", .9, (.31, .41, .41, .51)), firearm=True, source="firearm_model")
        f2 = dict(H.det("handgun", .7, (.32, .42, .42, .52)), firearm=True, source="firearm_model")  # duplicate of f1
        m = F.merge(base, [f1, f2])
        self.assertEqual(len(m), 3)                                      # 2 COCO kept + 1 firearm (dup removed)
        fire = [d for d in m if d.get("firearm")][0]
        self.assertEqual(fire["score"], .9); self.assertEqual(fire["overlaps_coco"], ["cell phone"])
        self.assertEqual({d["label"] for d in m}, {"person", "cell phone", "handgun"})   # COCO phone NOT suppressed

    def test_route_merges_and_context_marks_prediction(self):
        install([(0.91, 1, [60, 40, 80, 50])])
        j = self.up(self.img()).get_json()
        labels = [d["label"] for d in j["detections"]]
        self.assertEqual(sorted(labels), ["cell phone", "handgun", "person"])
        self.assertIn("handgun", [b["label"] for b in j["object_boxes"]["boxes"]])
        hg = [d for d in j["detections"] if d["label"] == "handgun"][0]
        self.assertEqual(hg["box"], {"x1": 0.3, "y1": 0.4, "x2": 0.4, "y2": 0.5}); self.assertEqual(hg["overlaps_coco"], ["cell phone"])
        ctx = fusion.build_context(appmod.load("fx"))
        self.assertIn("[FIREARM-MODEL prediction; overlaps COCO cell phone]", ctx)
        self.assertIn("separate firearm model", ctx)

    def test_firearm_failure_does_not_lose_coco_results(self):
        install([]); F.FIREARM_MODEL = "fake/firearm"
        with mock.patch.object(F, "detect", side_effect=F.FirearmModelError("model has no firearm-like class")):
            j = self.up(self.img()).get_json()
        self.assertEqual(j["firearm_detector"]["status"], "error"); self.assertIn("firearm", j["errors"])
        self.assertEqual(len(j["detections"]), 2)
        self.assertIn("FIREARM CHECK: FAILED", fusion.build_context(appmod.load("fx")))

    def test_bad_model_fails_loudly_on_load(self):
        F.FIREARM_MODEL = "fake/nofirearm"
        class M:
            class config: id2label = {0: "person", 1: "car"}
        fake_tf = type(sys)("transformers")
        fake_tf.AutoImageProcessor = type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
        fake_tf.AutoModelForObjectDetection = type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: type("X", (), {"eval": lambda s: M(), "config": M.config})())})
        with mock.patch.dict(sys.modules, {"transformers": fake_tf}):
            with self.assertRaises(F.FirearmModelError) as cm: F.detect(Image.new("RGB", (10, 10)))
        self.assertIn("no firearm-like class", str(cm.exception))


class VideoIntegration(Base):
    def mk(self, path):
        import cv2, numpy as np
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
        for i in range(25): w.write(np.full((120, 160, 3), 60, np.uint8))
        w.release()
    def test_firearm_events_and_summary_use_actual_timestamps(self):
        from pathlib import Path
        p = Path(H.tempfile.mkdtemp()) / "v.avi"; self.mk(p)
        hand = lambda img, base: base + [dict(H.det("handgun", .8, (.3, .3, .4, .4)), firearm=True, source="firearm_model")] if img is not None else base
        r = V.analyze_video_full(p, every_sec=1.0, detect_fn=lambda i: [H.det("person", .9)], firearm_fn=hand)
        ev = [e for e in r["events"] if e["type"] == "object_of_interest"]
        self.assertEqual(len(ev), 1); self.assertEqual(ev[0]["t"], 0.0)
        self.assertIn("firearm-model class; prediction only, not verified", ev[0]["text"])
        self.assertEqual(r["firearm"], {"ran": True, "frames_with_firearm_class": 3, "error": None})
        self.assertTrue(any(d["label"] == "handgun" for d in r["frames"][0]["detections"]))
    def test_firearm_failure_reported_once_and_video_continues(self):
        from pathlib import Path
        p = Path(H.tempfile.mkdtemp()) / "v.avi"; self.mk(p); calls = []
        def boom(img, base): calls.append(1); raise F.FirearmModelError("no firearm-like class")
        r = V.analyze_video_full(p, every_sec=1.0, detect_fn=lambda i: [H.det()], firearm_fn=boom)
        self.assertEqual(len(calls), 1); self.assertEqual(r["firearm"]["ran"], False); self.assertIn("no firearm-like class", r["errors"]["firearm"])
        self.assertEqual(r["summary"]["samples"], 3)
    def test_route_reports_not_configured(self):
        from pathlib import Path
        p = Path(H.tempfile.mkdtemp()) / "v.avi"; self.mk(p)
        j = self.up(p.read_bytes(), "v.avi", "fv", "video").get_json()
        self.assertEqual(j["firearm_detector"]["status"], "not_configured")
        self.assertIn("FIREARM CHECK: not run", fusion.build_context(appmod.load("fv")))


class LLMGuards(Base):
    def test_firearm_only_crime_label_is_downgraded(self):
        ids = {"ev_aaaaaaaaaa"}
        raw = {"category": "Robbery", "confidence": .9, "supporting_observations": [{"text": "A handgun was detected", "citations": ["ev_aaaaaaaaaa"]}]}
        out = G.validate_assessment(raw, ids)
        self.assertEqual(out["category"], "Insufficient evidence"); self.assertTrue(any("do not establish a crime" in n for n in out["validation_notes"]))
    def test_crime_label_with_other_cited_observation_is_kept(self):
        raw = {"category": "Robbery", "confidence": .6, "supporting_observations": [
            {"text": "A handgun was detected", "citations": ["ev_aaaaaaaaaa"]},
            {"text": "Audio segment says 'give me the money'", "citations": ["evt_12345678"]}]}
        self.assertEqual(G.validate_assessment(raw, {"ev_aaaaaaaaaa", "evt_12345678"})["category"], "Robbery")
    def test_prompt_contains_firearm_caveats(self):
        for t in ("NEVER establishes a crime", "lawful possession", "separate firearm model may or may not have been run"):
            self.assertIn(t, G._LIMITS)
    def test_llm_receives_firearm_prediction_with_caveat(self):
        install([(0.91, 1, [60, 40, 80, 50])]); G.GROQ_API_KEY = "k"
        seen = {}
        def fake(msgs, json_mode=False): seen["u"] = msgs[-1]["content"]; return json.dumps({"category": "Other", "confidence": .3, "supporting_observations": [{"text": "person", "citations": [appmod.load("fx")["images"][-1]["evidence_id"]]}]}) if False else json.dumps({"category": "Insufficient evidence"})
        with mock.patch.object(G, "_chat", side_effect=fake):
            self.up(self.img())
        self.assertIn("[FIREARM-MODEL prediction", seen["u"]); self.assertIn("NEVER establishes a crime", seen["u"])


class VerifyScript(unittest.TestCase):
    def test_flags_license_pickles_library_and_labels(self):
        good = VF.inspect({"license": "apache-2.0", "library_name": "transformers", "pipeline_tag": "object-detection",
                           "files": ["model.safetensors", "config.json"], "sha": "abc"}, {"architectures": ["DetrForObjectDetection"], "id2label": {"0": "gun"}}, F.select_labels)
        self.assertFalse([m for l, m in good if l == "FAIL"])
        bad = VF.inspect({"license": "agpl-3.0", "library_name": "ultralytics", "files": ["best.pt"]}, {"id2label": {"0": "car"}}, F.select_labels)
        lv = [l for l, _ in bad]; self.assertGreaterEqual(lv.count("FAIL"), 3); self.assertIn("WARN", lv)


if __name__ == "__main__":
    unittest.main()
