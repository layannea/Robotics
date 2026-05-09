import os
import re
from typing import Any, Dict, Optional


ROLE_MOVABLE = "movable_object"
ROLE_RECEPTACLE = "target_receptacle"
ROLE_BUTTON = "button_or_switch"
ROLE_PART = "handle_or_part"
ROLE_OBSTACLE = "obstacle"
ROLE_SURFACE = "surface_or_region"
ROLE_TOOL = "tool"
ROLE_UNKNOWN = "unknown"


CANONICAL_ROLES = {
    ROLE_MOVABLE,
    ROLE_RECEPTACLE,
    ROLE_BUTTON,
    ROLE_PART,
    ROLE_OBSTACLE,
    ROLE_SURFACE,
    ROLE_TOOL,
    ROLE_UNKNOWN,
}


def normalize_role(query_name: Optional[str], role_text: Optional[str]) -> str:
    query = (query_name or "").strip().lower().replace("_", " ")
    role = (role_text or "").strip().lower().replace("_", " ")
    text = f"{query} {role}"

    if any(token in query for token in ("button", "switch")):
        return ROLE_BUTTON
    if any(token in query for token in ("lid", "cap", "closure", "cover", "handle")):
        return ROLE_PART
    if any(token in query for token in ("target", "grill")):
        return ROLE_SURFACE
    if "scale" in query:
        return ROLE_SURFACE
    if any(token in query for token in ("bin", "stand", "saucepan", "bottle")):
        return ROLE_RECEPTACLE
    if "lamp" in query or "light" in query:
        return ROLE_UNKNOWN
    if any(
        token in query
        for token in (
            "rubbish",
            "trash",
            "paper",
            "umbrella",
            "meat",
            "pepper",
            "tomato",
            "block",
        )
    ):
        return ROLE_MOVABLE

    if any(token in role for token in ("button", "switch", "press", "push button")):
        return ROLE_BUTTON
    if any(token in role for token in ("lid", "cap", "closure", "cover", "handle", "part", "top")):
        return ROLE_PART
    if any(token in role for token in ("distractor", "obstacle", "not to", "remaining", "left on")):
        return ROLE_OBSTACLE
    if any(token in role for token in ("surface", "goal location", "target", "region", "place location")):
        return ROLE_SURFACE
    if any(token in role for token in ("receptacle", "container", "bin", "holder", "stand", "basket")):
        return ROLE_RECEPTACLE
    if any(
        token in role
        for token in (
            "movable",
            "item",
            "object to",
            "trash",
            "rubbish",
            "paper",
            "remove",
            "throw",
            "slide",
            "pick",
            "place into",
            "put",
            "umbrella",
            "meat",
            "pepper",
            "tomato",
            "block",
        )
    ):
        return ROLE_MOVABLE
    if "tool" in text:
        return ROLE_TOOL
    return ROLE_UNKNOWN


def object_entry_for_query(query_name: str, vlm_descriptions: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not vlm_descriptions:
        return None
    objects = (vlm_descriptions.get("parsed_output") or {}).get("objects") or {}
    entry = objects.get(query_name) or objects.get(query_name.lower())
    if entry is None:
        tomato_match = re.fullmatch(r"tomato\d*", query_name.lower())
        if tomato_match:
            entry = objects.get("tomato") or objects.get("tomato1")
    return entry if isinstance(entry, dict) else None


def role_for_query(query_name: str, vlm_descriptions: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    entry = object_entry_for_query(query_name, vlm_descriptions)
    raw_role = entry.get("role") if entry else None
    role_category = entry.get("role_category") if entry else None
    if role_category in CANONICAL_ROLES:
        normalized = role_category
    else:
        normalized = normalize_role(query_name, raw_role)
    return {
        "role": raw_role,
        "role_category": normalized,
    }


def query_name_from_phrase_candidates(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if parent.startswith("perception_"):
        return parent[len("perception_") :]
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return grandparent.split("_", 1)[0]


def vlm_summary_for_phrase_candidates(path: str) -> Optional[str]:
    cur = os.path.abspath(os.path.dirname(path))
    for _ in range(5):
        candidate = os.path.join(os.path.dirname(cur), "openai_vlm", "vlm_descriptions.json")
        if os.path.exists(candidate):
            return candidate
        direct = os.path.join(cur, "openai_vlm", "vlm_descriptions.json")
        if os.path.exists(direct):
            return direct
        cur = os.path.dirname(cur)
    return None
