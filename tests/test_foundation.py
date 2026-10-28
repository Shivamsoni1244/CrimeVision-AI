import io, pytest
import storage, validation
from app import app
from llm.grounding import extract
from pipeline.fusion import build_timeline, build_context

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
def post(kind, data, name, case="c1"):
    return app.test_client().post(f"/api/evidence/{kind}", data={"file": (io.BytesIO(data), name), "case_id": case})

def test_rejects_wrong_content_and_extension():
    assert post("image", b"not an image at all", "a.png").status_code == 400
    assert post("image", PNG, "a.exe").status_code == 400
    with pytest.raises(validation.ValidationError): validation.check("video", "a.mp4", PNG)

def test_ingest_hash_and_tamper_detection():
    r = post("image", PNG, "../../evil.png").get_json()
    ev = r["evidence"]
    assert len(ev["sha256"]) == 64 and ".." not in ev["path"]
    assert all(o["ok"] for o in app.test_client().get("/api/integrity/c1").get_json())
    open(ev["path"], "ab").write(b"x")
    res = app.test_client().get("/api/integrity/c1").get_json()
    assert any(not o["ok"] and o["reason"] == "hash mismatch" for o in res)
    acts = [a["action"] for a in app.test_client().get("/api/audit/c1").get_json()]
    assert "ingest" in acts and "integrity_check" in acts

def test_bad_case_id_and_unknown_case():
    assert post("image", PNG, "a.png", case="../x").status_code == 400
    assert app.test_client().get("/api/timeline/nope").status_code == 404

def test_timeline_untimed_images_last_and_ids_stable():
    case = {"id": "t", "images": [{"evidence_id": "ev_aaaaaaaaaa", "file": "i.png", "caption": "a room",
            "detections": [], "scene": [{"label": "calm", "score": .9}]}],
            "video_events": [{"t": 5, "source": "video", "type": "scene", "text": "v", "evidence_id": "ev_bbbbbbbbbb"}],
            "audio": [{"evidence_id": "ev_cccccccccc", "segments": [{"start": 2.0, "end": 3, "text": "hi"}]}]}
    tl = build_timeline(case)
    assert [e["source"] for e in tl] == ["audio", "video", "image"] and tl[2]["t"] is None
    assert tl[0]["event_id"] == build_timeline(case)[0]["event_id"]
    assert "untimed" in build_context(case)

def test_citation_check_flags_fabricated_ids():
    r = extract("Seen [evt_12345678] and [evt_deadbeef].", {"evt_12345678"})
    assert r == {"valid": ["evt_12345678"], "unknown": ["evt_deadbeef"]}
