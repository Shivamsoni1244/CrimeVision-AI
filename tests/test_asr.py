import pytest
from models import asr


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return {"language": "hi", "text": " नमस्ते ", "segments": [
            {"start": -0.2, "end": 1.256, "text": " नमस्ते "},
            {"start": 2, "end": 1, "text": "invalid segment"},
        ]}


def test_transcribe_preserves_language_and_normalizes_segments(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(asr, "_load_model", lambda: model)
    result = asr.transcribe("sample.wav")
    assert model.calls[0][1]["task"] == "transcribe"
    assert result["language"] == "hi"
    assert result["translated_to"] is None
    assert result["segments"] == [{"start": 0.0, "end": 1.26, "text": "नमस्ते"}]


def test_translate_mode_and_invalid_task(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(asr, "_load_model", lambda: model)
    result = asr.transcribe("sample.wav", task="translate", language="hi")
    assert model.calls[0][1]["task"] == "translate"
    assert model.calls[0][1]["language"] == "hi"
    assert result["translated_to"] == "en"
    with pytest.raises(ValueError):
        asr.transcribe("sample.wav", task="summarize")
