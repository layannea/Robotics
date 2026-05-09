import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .role_aware import (
        ROLE_BUTTON,
        ROLE_MOVABLE,
        ROLE_PART,
        ROLE_RECEPTACLE,
        ROLE_SURFACE,
        normalize_role,
        query_name_from_phrase_candidates,
        role_for_query,
        vlm_summary_for_phrase_candidates,
    )
except ImportError:
    from role_aware import (
        ROLE_BUTTON,
        ROLE_MOVABLE,
        ROLE_PART,
        ROLE_RECEPTACLE,
        ROLE_SURFACE,
        normalize_role,
        query_name_from_phrase_candidates,
        role_for_query,
        vlm_summary_for_phrase_candidates,
    )


_QUERY_DIR_RE = re.compile(r"^(?P<object>.+)_\d{10,}_(?P<obs_id>\d+)$")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_dir_for_summary(summary_path: str) -> Optional[str]:
    cur = os.path.abspath(os.path.dirname(summary_path))
    for _ in range(5):
        if os.path.basename(cur).startswith("perception_"):
            return os.path.dirname(cur)
        cur = os.path.dirname(cur)
    return None


def _query_dir_parts(query_dir: str) -> Optional[Dict[str, str]]:
    match = _QUERY_DIR_RE.match(os.path.basename(query_dir))
    if not match:
        return None
    return match.groupdict()


def _selected_summary_from_phrase_candidates(path: str) -> Optional[str]:
    try:
        data = _load_json(path)
    except Exception:
        return None
    for record in data.get("records") or []:
        if record.get("selected") and record.get("summary") and os.path.exists(record["summary"]):
            return record["summary"]
    selected = data.get("selected_replay_summary")
    if selected and os.path.exists(selected):
        return selected
    return None


def _role_info_from_phrase_candidates(path: str, object_name: str) -> Dict[str, Optional[str]]:
    vlm_path = vlm_summary_for_phrase_candidates(path)
    vlm_summary = _load_json(vlm_path) if vlm_path and os.path.exists(vlm_path) else None
    return role_for_query(query_name_from_phrase_candidates(path) or object_name, vlm_summary)


def sibling_selected_summaries(summary_path: str) -> List[Dict[str, str]]:
    query_dir = _query_dir_for_summary(summary_path)
    if not query_dir:
        return []
    parts = _query_dir_parts(query_dir)
    if not parts:
        return []
    task_dir = os.path.dirname(query_dir)
    siblings = []
    for candidate_dir in glob.glob(os.path.join(task_dir, f"*_{parts['obs_id']}")):
        if os.path.abspath(candidate_dir) == os.path.abspath(query_dir):
            continue
        candidate_parts = _query_dir_parts(candidate_dir)
        if not candidate_parts:
            continue
        for phrase_candidates in glob.glob(
            os.path.join(candidate_dir, "perception_*", "phrase_candidates.json")
        ):
            selected_summary = _selected_summary_from_phrase_candidates(phrase_candidates)
            if selected_summary:
                role_info = _role_info_from_phrase_candidates(phrase_candidates, candidate_parts["object"])
                siblings.append(
                    {
                        "object": candidate_parts["object"],
                        "role": role_info.get("role"),
                        "role_category": role_info.get("role_category"),
                        "phrase_candidates": phrase_candidates,
                        "summary": selected_summary,
                    }
                )
                break
    return sorted(siblings, key=lambda item: (item["object"], item["summary"]))


def distractor_masks_by_camera(summary_path: str) -> Dict[str, List[Dict[str, str]]]:
    masks: Dict[str, List[Dict[str, str]]] = {}
    for sibling in sibling_selected_summaries(summary_path):
        try:
            summary = _load_json(sibling["summary"])
        except Exception:
            continue
        for result in summary.get("camera_results") or []:
            if not result.get("used_for_fusion"):
                continue
            mask_path = result.get("mask")
            camera = result.get("camera")
            if not camera or not mask_path or not os.path.exists(mask_path):
                continue
            masks.setdefault(camera, []).append(
                {
                    "object": sibling["object"],
                    "role": sibling.get("role"),
                    "role_category": sibling.get("role_category"),
                    "summary": sibling["summary"],
                    "mask": mask_path,
                    "object_point_center": result.get("object_point_center"),
                    "object_point_count": result.get("object_point_count"),
                }
            )
    return masks


def mask_iou(mask_a_path: Optional[str], mask_b_path: Optional[str]) -> Optional[float]:
    if not mask_a_path or not mask_b_path:
        return None
    if not os.path.exists(mask_a_path) or not os.path.exists(mask_b_path):
        return None
    mask_a = np.load(mask_a_path).astype(bool)
    mask_b = np.load(mask_b_path).astype(bool)
    if mask_a.shape != mask_b.shape:
        mask_a = mask_a.reshape(-1)
        mask_b = mask_b.reshape(-1)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return None
    return float(intersection / union)


def _as_center(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return arr


def _center_dist(a: Any, b: Any) -> Optional[float]:
    center_a = _as_center(a)
    center_b = _as_center(b)
    if center_a is None or center_b is None:
        return None
    return float(np.linalg.norm(center_a - center_b))


def _name_has(name: Optional[str], tokens: List[str]) -> bool:
    text = (name or "").lower().replace("_", " ")
    return any(token in text for token in tokens)


def relation_between(
    query_object: Optional[str],
    query_role: Optional[str],
    distractor_object: Optional[str],
    distractor_role: Optional[str],
) -> str:
    query_role = query_role or normalize_role(query_object, None)
    distractor_role = distractor_role or normalize_role(distractor_object, None)

    if query_role in (ROLE_PART, ROLE_BUTTON):
        if (
            (_name_has(query_object, ["cap"]) and _name_has(distractor_object, ["bottle"]))
            or (_name_has(query_object, ["lid"]) and _name_has(distractor_object, ["saucepan", "pot"]))
            or (_name_has(query_object, ["button", "switch"]) and _name_has(distractor_object, ["lamp"]))
            or (_name_has(query_object, ["handle"]) and _name_has(distractor_object, ["umbrella", "saucepan", "pot", "bottle"]))
        ):
            return "attached_part_parent"
        if distractor_role in (ROLE_RECEPTACLE, ROLE_SURFACE):
            return "attached_part_parent"

    if query_role in (ROLE_RECEPTACLE, ROLE_SURFACE) and distractor_role in (ROLE_MOVABLE, ROLE_PART, ROLE_BUTTON):
        return "support_or_container"
    if query_role == ROLE_MOVABLE and distractor_role in (ROLE_RECEPTACLE, ROLE_SURFACE):
        return "target_contact"
    if query_role == distractor_role and query_role in (ROLE_MOVABLE, ROLE_PART, ROLE_BUTTON, ROLE_RECEPTACLE, ROLE_SURFACE):
        return "peer_competitor"
    if query_role == ROLE_MOVABLE and distractor_role == ROLE_MOVABLE:
        return "peer_competitor"
    return "ambiguous"


def annotate_candidate_with_distractors(
    candidate: Dict[str, Any],
    distractors_by_camera: Dict[str, List[Dict[str, str]]],
    *,
    query_object: Optional[str] = None,
    query_role: Optional[str] = None,
) -> Dict[str, Any]:
    mask_path = candidate.get("mask")
    camera = candidate.get("camera")
    overlaps = []
    for distractor in distractors_by_camera.get(camera, []):
        iou = mask_iou(mask_path, distractor.get("mask"))
        if iou is None:
            continue
        center_dist = _center_dist(candidate.get("object_point_center"), distractor.get("object_point_center"))
        relation = relation_between(
            query_object=query_object,
            query_role=query_role,
            distractor_object=distractor.get("object"),
            distractor_role=distractor.get("role_category"),
        )
        overlaps.append(
            {
                "object": distractor.get("object"),
                "role_category": distractor.get("role_category"),
                "relation": relation,
                "summary": distractor.get("summary"),
                "mask": distractor.get("mask"),
                "center_dist_m": center_dist,
                "iou": iou,
            }
        )
    overlaps.sort(key=lambda item: item["iou"], reverse=True)
    negative_overlaps = [
        item
        for item in overlaps
        if item["relation"] == "peer_competitor"
        and item["iou"] > 0.10
        and (item["center_dist_m"] is None or item["center_dist_m"] < 0.12)
    ]
    negative_overlaps.sort(key=lambda item: (item["iou"], -(item["center_dist_m"] or 0.0)), reverse=True)
    candidate["distractor_overlaps"] = overlaps
    candidate["max_distractor_iou"] = overlaps[0]["iou"] if overlaps else 0.0
    candidate["max_distractor_object"] = overlaps[0]["object"] if overlaps else None
    candidate["max_distractor_relation"] = overlaps[0]["relation"] if overlaps else None
    candidate["negative_distractor_overlaps"] = negative_overlaps
    candidate["max_negative_distractor_iou"] = negative_overlaps[0]["iou"] if negative_overlaps else 0.0
    candidate["max_negative_distractor_object"] = negative_overlaps[0]["object"] if negative_overlaps else None
    candidate["max_negative_distractor_center_dist_m"] = (
        negative_overlaps[0]["center_dist_m"] if negative_overlaps else None
    )
    return candidate
