"""SQLite evidence registry, SHA-256 integrity, audit log. Originals are never modified."""
import hashlib, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from config import DATA
import validation

DB = DATA / "crimevision.db"
def _c():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def _now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

def init_db():
    DATA.mkdir(parents=True, exist_ok=True)
    with _c() as c:
        c.executescript("""CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, case_id TEXT, kind TEXT,
          filename TEXT, path TEXT, sha256 TEXT, size INTEGER, ingested_at TEXT);
          CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, case_id TEXT,
          action TEXT, detail TEXT);""")

def audit(case_id, action, detail=""):
    with _c() as c:
        c.execute("INSERT INTO audit(ts,case_id,action,detail) VALUES(?,?,?,?)", (_now(), case_id, action, detail))

def ingest(case_id, kind, f):
    name = f.filename or ""
    head = f.stream.read(32); f.stream.seek(0)
    try:
        validation.check(kind, name, head)
    except validation.ValidationError as e:
        audit(case_id, "ingest_rejected", f"{name}: {e}"); raise
    eid = "ev_" + uuid.uuid4().hex[:10]
    d = DATA / case_id / "originals"; d.mkdir(parents=True, exist_ok=True)
    path = d / (eid + Path(name).suffix.lower())
    h, size = hashlib.sha256(), 0
    with open(path, "wb") as out:
        while chunk := f.stream.read(1 << 20):
            h.update(chunk); out.write(chunk); size += len(chunk)
    rec = {"id": eid, "case_id": case_id, "kind": kind, "filename": name, "path": str(path),
           "sha256": h.hexdigest(), "size": size, "ingested_at": _now()}
    with _c() as c:
        c.execute("INSERT INTO evidence VALUES(:id,:case_id,:kind,:filename,:path,:sha256,:size,:ingested_at)", rec)
    audit(case_id, "ingest", f"{eid} {name} sha256={rec['sha256']}")
    return rec

def list_evidence(case_id):
    with _c() as c:
        return [dict(r) for r in c.execute("SELECT * FROM evidence WHERE case_id=? ORDER BY ingested_at", (case_id,))]

def verify(case_id):
    out = []
    for r in list_evidence(case_id):
        p = Path(r["path"])
        if not p.exists():
            out.append({"evidence_id": r["id"], "ok": False, "reason": "file missing"}); continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while chunk := fh.read(1 << 20): h.update(chunk)
        out.append({"evidence_id": r["id"], "ok": h.hexdigest() == r["sha256"],
                    "reason": "" if h.hexdigest() == r["sha256"] else "hash mismatch"})
    audit(case_id, "integrity_check", f"{sum(o['ok'] for o in out)}/{len(out)} ok")
    return out

def get_audit(case_id):
    with _c() as c:
        return [dict(r) for r in c.execute("SELECT ts,action,detail FROM audit WHERE case_id=? ORDER BY id", (case_id,))]
