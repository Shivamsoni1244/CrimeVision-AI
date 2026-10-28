"""Check that cited record IDs in an LLM answer actually exist."""
import re
_ID = re.compile(r"\bev(?:t)?_[0-9a-f]{8,10}\b")

def extract(answer: str, valid_ids: set) -> dict:
    found = list(dict.fromkeys(_ID.findall(answer)))
    return {"valid": [i for i in found if i in valid_ids], "unknown": [i for i in found if i not in valid_ids]}
