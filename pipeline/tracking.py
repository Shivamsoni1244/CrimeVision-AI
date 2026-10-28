"""Minimal IoU/centre-distance multi-object tracker over SAMPLED frames (no extra dependencies).

This associates detections of the same class between consecutive samples. It is NOT re-identification:
at 1 s sampling, fast-moving objects may be split into several tracks and crossing objects may swap ids.
Track ids are labels for "the same-looking box continued", not identities of people.
"""
from dataclasses import dataclass, field


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def centre(b: dict):
    return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)


@dataclass
class Track:
    id: int
    label: str
    first_t: float
    first_frame: int
    last_t: float
    last_frame: int
    box: dict
    missed: int = 0
    n_obs: int = 1
    max_conf: float = 0.0
    path: list = field(default_factory=list)  # [(t, cx, cy)]

    def to_dict(self):
        return {"track_id": f"T{self.id}", "label": self.label, "first_t": self.first_t, "last_t": self.last_t,
                "first_frame": self.first_frame, "last_frame": self.last_frame, "observations": self.n_obs,
                "max_confidence": self.max_conf,
                "path": [{"t": t, "cx": round(x, 3), "cy": round(y, 3)} for t, x, y in self.path]}


class IoUTracker:
    def __init__(self, iou_min=0.2, dist_max=0.3, max_missed=2):
        self.iou_min, self.dist_max, self.max_missed = iou_min, dist_max, max_missed
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    def update(self, t: float, frame: int, dets: list):
        """dets: [{label, confidence, box{x1..y2}}]. Returns [(det, Track, is_new)] for this sample."""
        pairs = []
        for di, d in enumerate(dets):
            for tr in self.active:
                if tr.label != d["label"]:
                    continue
                o = iou(tr.box, d["box"])
                (cx, cy), (tx, ty) = centre(d["box"]), centre(tr.box)
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if o >= self.iou_min or dist <= self.dist_max:
                    pairs.append((o - dist * 0.1, di, tr))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_d, used_t, out = set(), set(), []
        for _, di, tr in pairs:
            if di in used_d or tr.id in used_t:
                continue
            used_d.add(di); used_t.add(tr.id)
            d = dets[di]
            tr.box, tr.last_t, tr.last_frame, tr.missed = d["box"], t, frame, 0
            tr.n_obs += 1; tr.max_conf = max(tr.max_conf, d["confidence"])
            tr.path.append((t, *centre(d["box"])))
            out.append((d, tr, False))
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(self._next, d["label"], t, frame, t, frame, d["box"], max_conf=d["confidence"],
                       path=[(t, *centre(d["box"]))])
            self._next += 1
            self.active.append(tr)
            out.append((d, tr, True))
        for tr in list(self.active):
            if tr.id not in used_t and not any(tr is o[1] for o in out):
                tr.missed += 1
                if tr.missed > self.max_missed:
                    self.active.remove(tr); self.finished.append(tr)
        return out

    def all_tracks(self):
        return sorted(self.finished + self.active, key=lambda t: t.id)
