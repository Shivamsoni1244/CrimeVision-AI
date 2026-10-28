"""CLIP zero-shot scene/weapon classification + BLIP captions.

Fix: the old _load() guarded everything on `_clip is None`. If CLIP loaded but BLIP failed
(download error, offline, out of memory), `_clip` stayed set, so every later call skipped
loading and `_blip_p` remained None -> "TypeError: 'NoneType' object is not callable".
Each model is now loaded independently and published only after BOTH its model and
processor loaded, so a failure is retried on the next request and the real error surfaces.
"""
import threading

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

SCENES = ["a robbery in progress", "a physical assault or fight", "a traffic accident",
          "a person holding a handgun", "a person holding a knife", "a normal peaceful scene",
          "a fire or explosion", "a burglary or break-in"]

_CLIP_ID = "openai/clip-vit-base-patch32"
_BLIP_ID = "Salesforce/blip-image-captioning-base"
_lock = threading.Lock()
_clip = None  # (model, processor) or None
_blip = None  # (model, processor) or None


def _load_clip():
    global _clip
    if _clip is None:
        with _lock:
            if _clip is None:
                model = CLIPModel.from_pretrained(_CLIP_ID).eval()
                proc = CLIPProcessor.from_pretrained(_CLIP_ID)
                _clip = (model, proc)
    return _clip


def _load_blip():
    global _blip
    if _blip is None:
        with _lock:
            if _blip is None:
                model = BlipForConditionalGeneration.from_pretrained(_BLIP_ID).eval()
                proc = BlipProcessor.from_pretrained(_BLIP_ID)
                _blip = (model, proc)
    return _blip


def classify_scene(image: Image.Image, top_k: int = 3):
    clip, clip_p = _load_clip()
    inp = clip_p(text=SCENES, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip(**inp).logits_per_image.softmax(dim=1)[0]
    top = probs.topk(top_k)
    return [{"label": SCENES[i], "score": round(float(s), 3)} for s, i in zip(top.values, top.indices)]


def caption(image: Image.Image) -> str:
    blip, blip_p = _load_blip()
    inp = blip_p(image, return_tensors="pt")
    with torch.no_grad():
        out = blip.generate(**inp, max_new_tokens=40)
    return blip_p.decode(out[0], skip_special_tokens=True)
