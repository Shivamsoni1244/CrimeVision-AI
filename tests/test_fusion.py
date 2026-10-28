from pipeline.fusion import build_timeline

def test_timeline_sorted_across_modalities():
    case = {"id": "t", "video_events": [{"t": 5, "source": "video", "type": "scene", "text": "v"}],
            "audio": [{"segments": [{"start": 2.0, "end": 3, "text": "a"}]}], "images": []}
    assert [e["source"] for e in build_timeline(case)] == ["audio", "video"]
