"""Check a candidate firearm-detection model BEFORE enabling it.   Usage (project root, internet needed):

    python scripts/verify_firearm_model.py <huggingface-model-id>

Prints: license, library, weight formats, gating, commit hash to pin, the model's class labels, and which classes
would count as firearms. It downloads only config.json (a few KB), never the weights. It cannot judge accuracy:
you must still test the model on your own positive and negative images and read its model card/training data.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "cc0-1.0", "openrail", "openrail++"}


def inspect(info: dict, config: dict, select_labels) -> list:
    """Pure function (testable offline). Returns [(level, message)] with level in OK/WARN/FAIL."""
    out = []
    lic = (info.get("license") or "").lower()
    if not lic:
        out.append(("WARN", "No license declared on the model card. Do not use until you know the terms."))
    elif lic in PERMISSIVE:
        out.append(("OK", f"License: {lic} (permissive; still read the card for dataset/use restrictions)."))
    else:
        out.append(("WARN", f"License: {lic}. Check that it allows your use (non-commercial / copyleft terms?)."))
    if info.get("gated"):
        out.append(("WARN", "Model is gated: you must accept its terms on huggingface.co and log in first."))
    files = info.get("files") or []
    if any(f.endswith(".safetensors") for f in files):
        out.append(("OK", "Has .safetensors weights (the project loads with use_safetensors=True)."))
    else:
        out.append(("FAIL", "No .safetensors weights; pickle-based files (.bin/.pt/.pkl) can run code on load."))
    if any(f.endswith((".pt", ".pkl", ".pth")) and not f.endswith(".safetensors") for f in files):
        out.append(("WARN", "Repository also contains pickle-style files; they are not loaded, but be aware."))
    if info.get("library_name") not in (None, "transformers"):
        out.append(("FAIL", f"library_name is '{info.get('library_name')}', not transformers (e.g. a YOLO/ultralytics "
                            "checkpoint will not load with AutoModelForObjectDetection)."))
    if info.get("pipeline_tag") not in (None, "object-detection"):
        out.append(("FAIL", f"pipeline_tag is '{info.get('pipeline_tag')}', expected object-detection."))
    archs = config.get("architectures") or []
    out.append(("OK" if archs else "WARN", f"Architectures: {archs or 'not listed'}."))
    id2label = config.get("id2label") or {}
    out.append(("OK" if id2label else "FAIL", f"Classes ({len(id2label)}): {sorted(set(map(str, id2label.values())))[:30]}"))
    try:
        chosen = select_labels(id2label)
        out.append(("OK", f"Classes that will count as firearms: {sorted(chosen.values())}. Confirm these names mean what you expect."))
    except Exception as e:
        out.append(("FAIL", str(e)))
    if info.get("sha"):
        out.append(("OK", f"Pin this commit in .env: FIREARM_REVISION={info['sha']}"))
    out.append(("WARN", "Accuracy is NOT verified by this script. Read the model card (training data, metrics) and test on your own positive and negative images."))
    return out


def main(model_id: str):
    from huggingface_hub import HfApi, hf_hub_download
    from models.firearm import select_labels
    mi = HfApi().model_info(model_id)
    cd = getattr(mi, "card_data", None)
    info = {"license": (getattr(cd, "license", None) if cd else None) or next((t.split(":", 1)[1] for t in mi.tags if t.startswith("license:")), None),
            "gated": bool(getattr(mi, "gated", False)), "library_name": mi.library_name, "pipeline_tag": mi.pipeline_tag,
            "files": [s.rfilename for s in mi.siblings or []], "sha": mi.sha}
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    print(f"Model: {model_id}")
    bad = False
    for lvl, msg in inspect(info, cfg, select_labels):
        print(f"[{lvl}] {msg}"); bad |= lvl == "FAIL"
    print("\nRESULT:", "NOT suitable as-is (fix the FAIL items)." if bad else "No blocking problems found; now test on your own images before relying on it.")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
