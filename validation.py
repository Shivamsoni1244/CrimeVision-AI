"""Upload validation: allowed extension AND matching file signature."""
from pathlib import Path
EXT = {"image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"},
       "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
       "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}}

class ValidationError(ValueError):
    pass

def sniff(h: bytes) -> set:
    k = set()
    if h.startswith(b"\x89PNG\r\n\x1a\n") or h[:3] == b"\xff\xd8\xff" or h[:4] == b"GIF8" \
            or h[:2] == b"BM" or (h[:4] == b"RIFF" and h[8:12] == b"WEBP"):
        k.add("image")
    if h[4:8] == b"ftyp" or h[:4] == b"\x1a\x45\xdf\xa3":
        k |= {"video", "audio"}
    if h[:4] == b"RIFF" and h[8:12] == b"AVI ":
        k.add("video")
    if (h[:4] == b"RIFF" and h[8:12] == b"WAVE") or h[:3] == b"ID3" or h[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") \
            or h[:4] in (b"OggS", b"fLaC"):
        k.add("audio")
    return k

def check(kind: str, filename: str, head: bytes):
    if kind not in EXT:
        raise ValidationError(f"unknown evidence kind '{kind}'")
    if Path(filename or "").suffix.lower() not in EXT[kind]:
        raise ValidationError(f"extension not allowed for {kind}: {sorted(EXT[kind])}")
    if kind not in sniff(head):
        raise ValidationError(f"file content does not look like a valid {kind} file")
