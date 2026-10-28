# Project Audit (updated after foundation pass)
Environment note: Hugging Face is unreachable from the build sandbox, so **no model inference (RT-DETR, CLIP, BLIP, Whisper) has been executed or verified**. Groq was not called.

| Area | Status |
|---|---|
| Flask API, case JSON store, dashboard, PDF skeleton | Implemented, only partly tested (API ingest paths tested; model routes unverified) |
| RT-DETR / CLIP / BLIP / Whisper / frame sampling | Implemented, **unverified** (code compiles only) |
| Upload validation (extension + magic bytes + size cap) | Implemented, tested |
| SHA-256 at ingestion, originals preserved, integrity check, SQLite audit log | Implemented, tested |
| Stable event IDs, review_status, untimed image observations | Implemented, tested |
| Evidence chat: untrusted-data framing, citations, insufficient-evidence rule, citation check | Implemented; prompt logic unverified against Groq, citation checker tested |
| Firearm detection | **Blocked**: COCO RT-DETR has no firearm class; needs a licensed weapons model/weights |
| Object tracking, action recognition, diarization, scene-change detection | **Not implemented** |
| Annotated images/frames, transcript correction, investigator review UI/API | **Not implemented** |
| Auth, CSRF, rate limiting, retention/deletion, background jobs + progress | **Not implemented** (server is unauthenticated; do not expose it) |
| PDF: provenance, hashes, limitations, Unicode/large-data tests | **Not implemented** (original generator only) |
| Groq retries/timeouts/JSON schema validation, context retrieval for large cases | **Not implemented** |
| 3D view | 2D-derived layout only; no real depth or distances |
Defects found and fixed: image evidence was placed at t=0 on the timeline (now untimed); user filenames were used as paths (now stored as `<evidence_id><ext>`); `debug=True` was hard-coded (now `FLASK_DEBUG=1` only); chat gave no citations.
