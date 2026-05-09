import math
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .role_aware import (
        ROLE_BUTTON,
        ROLE_MOVABLE,
        ROLE_OBSTACLE,
        ROLE_PART,
        ROLE_RECEPTACLE,
        ROLE_SURFACE,
        normalize_role,
    )
except ImportError:
    from role_aware import (
        ROLE_BUTTON,
        ROLE_MOVABLE,
        ROLE_OBSTACLE,
        ROLE_PART,
        ROLE_RECEPTACLE,
        ROLE_SURFACE,
        normalize_role,
    )


SCORER_VERSION = "v2.0"
ROLE_VERIFIER_MODE = "role_verifier_mode"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _workspace_soft_max_z(manifest: Dict[str, Any], margin: float = 0.75) -> Optional[float]:
    bounds_min = manifest.get("workspace_bounds_min")
    if not bounds_min:
        return None
    return float(bounds_min[2]) + margin


def _box_touches_edge(box, image_shape, margin: float = 1.0) -> bool:
    if box is None or image_shape is None:
        return False
    height, width = float(image_shape[0]), float(image_shape[1])
    return (
        float(box[0]) <= margin
        or float(box[1]) <= margin
        or float(box[2]) >= width - margin
        or float(box[3]) >= height - margin
    )


def _finite_centers(camera_results: List[Dict[str, Any]]) -> np.ndarray:
    centers = []
    for result in camera_results:
        center = result.get("object_point_center")
        if center is None:
            continue
        arr = np.asarray(center, dtype=np.float32)
        if arr.shape == (3,) and np.isfinite(arr).all():
            centers.append(arr)
    if not centers:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(centers, axis=0)


def _mean_pairwise_distance(points: np.ndarray) -> Optional[float]:
    if len(points) < 2:
        return None
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(float(np.linalg.norm(points[i] - points[j])))
    return float(np.mean(distances)) if distances else None


def _role(config: Dict[str, Any]) -> str:
    return config.get("role_category") or normalize_role(config.get("query_name"), config.get("role"))


def _role_verifier_mode(config: Dict[str, Any]) -> str:
    return str(config.get(ROLE_VERIFIER_MODE) or "soft")


def _apply_camera_role_prior(
    score: float,
    reasons: List[str],
    role: str,
    candidate: Dict[str, Any],
    trigger_points: float,
    trigger_box_area_frac: float,
) -> float:
    point_count = _as_int(candidate.get("object_point_count"), 0)
    box_area_frac = candidate.get("box_area_frac")
    box_area_frac = _as_float(box_area_frac, 0.0) if box_area_frac is not None else None

    if role == ROLE_MOVABLE:
        if point_count > 1.4 * trigger_points:
            score -= 0.55
            reasons.append("role_movable_large_points")
        if box_area_frac is not None and box_area_frac > trigger_box_area_frac:
            score -= 0.35
            reasons.append("role_movable_large_box")
    elif role == ROLE_RECEPTACLE:
        if point_count > trigger_points:
            score += 0.35
            reasons.append("role_receptacle_allows_large_region")
        if box_area_frac is not None and box_area_frac < 0.01:
            score -= 0.45
            reasons.append("role_receptacle_too_tiny")
    elif role == ROLE_BUTTON:
        if point_count <= 2000:
            score += 0.35
            reasons.append("role_button_small_region")
        if box_area_frac is not None and box_area_frac > 0.08:
            score -= 0.65
            reasons.append("role_button_region_too_large")
    elif role == ROLE_PART:
        if point_count <= trigger_points:
            score += 0.25
            reasons.append("role_part_local_region")
        if point_count > 1.8 * trigger_points:
            score -= 0.65
            reasons.append("role_part_too_large")
    elif role == ROLE_SURFACE:
        if point_count > trigger_points:
            score += 0.25
            reasons.append("role_surface_allows_broad_region")
    elif role == ROLE_OBSTACLE:
        if point_count > 1.5 * trigger_points:
            score -= 0.35
            reasons.append("role_obstacle_large_region")
    return score


def _apply_phrase_role_prior(
    score: float,
    reasons: List[str],
    role: str,
    fused_count: int,
    used: List[Dict[str, Any]],
    max_points: Optional[Any],
) -> float:
    mean_box_area = None
    areas = [result.get("box_area_frac") for result in used if result.get("box_area_frac") is not None]
    if areas:
        mean_box_area = float(np.mean([_as_float(area) for area in areas]))

    if role == ROLE_MOVABLE:
        if max_points is not None and fused_count > 0.8 * _as_int(max_points, fused_count):
            score -= 0.45
            reasons.append("role_movable_near_max_points")
        if mean_box_area is not None and mean_box_area > 0.18:
            score -= 0.45
            reasons.append("role_movable_large_view_area")
    elif role == ROLE_RECEPTACLE:
        if fused_count >= 2500:
            score += 0.45
            reasons.append("role_receptacle_large_support")
        if fused_count < 500:
            score -= 0.35
            reasons.append("role_receptacle_low_support")
    elif role == ROLE_BUTTON:
        if fused_count <= 2500:
            score += 0.45
            reasons.append("role_button_compact")
        if fused_count > 4500:
            score -= 0.75
            reasons.append("role_button_too_large")
    elif role == ROLE_PART:
        if 150 <= fused_count <= 5000:
            score += 0.35
            reasons.append("role_part_reasonable_extent")
        if fused_count > 9000:
            score -= 0.65
            reasons.append("role_part_too_large")
    elif role == ROLE_SURFACE:
        if fused_count >= 1000:
            score += 0.25
            reasons.append("role_surface_region_support")
    elif role == ROLE_OBSTACLE:
        if fused_count > 7000:
            score -= 0.35
            reasons.append("role_obstacle_large_extent")
    return score


def _apply_strict_role_verifier(
    score: float,
    reasons: List[str],
    role: str,
    fused_count: int,
    used: List[Dict[str, Any]],
    centers: np.ndarray,
) -> float:
    used_count = len(used)
    mean_pairwise = _mean_pairwise_distance(centers)
    if role == ROLE_MOVABLE:
        if used_count < 2:
            score -= 0.85
            reasons.append("strict_movable_requires_multiview")
        if fused_count > 4500:
            score -= 0.7
            reasons.append("strict_movable_too_large")
        if mean_pairwise is not None and mean_pairwise > 0.12:
            score -= 0.7
            reasons.append(f"strict_movable_inconsistent={mean_pairwise:.3f}")
    elif role == ROLE_BUTTON:
        if fused_count > 3500:
            score -= 1.0
            reasons.append("strict_button_too_large")
    elif role == ROLE_PART:
        if fused_count > 6500:
            score -= 0.8
            reasons.append("strict_part_too_large")
    elif role == ROLE_RECEPTACLE:
        if fused_count < 350:
            score -= 0.7
            reasons.append("strict_receptacle_too_small")
    elif role == ROLE_SURFACE:
        if used_count < 2:
            score -= 0.45
            reasons.append("strict_surface_prefers_multiview")
        if mean_pairwise is not None and mean_pairwise > 0.18:
            score -= 0.8
            reasons.append(f"strict_surface_inconsistent={mean_pairwise:.3f}")
    return score


def _apply_distractor_penalty(score: float, reasons: List[str], role: str, candidate: Dict[str, Any]) -> float:
    iou = _as_float(candidate.get("max_negative_distractor_iou"), 0.0)
    if iou <= 0.0:
        return score

    distractor_object = candidate.get("max_negative_distractor_object") or "unknown"
    center_dist = candidate.get("max_negative_distractor_center_dist_m")
    center_dist = _as_float(center_dist, -1.0) if center_dist is not None else None
    penalty = 0.0
    if role in (ROLE_MOVABLE, ROLE_PART, ROLE_BUTTON, ROLE_RECEPTACLE, ROLE_SURFACE):
        if iou > 0.50:
            penalty = 1.35
        elif iou > 0.25:
            penalty = 0.75
        elif iou > 0.10:
            penalty = 0.30
    else:
        return score

    if center_dist is not None and center_dist > 0.08:
        penalty *= 0.55
    if penalty > 0.0:
        score -= penalty
        if center_dist is None:
            reasons.append(f"negative_peer_iou={iou:.2f}:{distractor_object}")
        else:
            reasons.append(f"negative_peer_iou={iou:.2f}:{distractor_object}:dist={center_dist:.3f}")
    return score


def score_camera_candidate_v2(
    candidate: Dict[str, Any],
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    trigger_points: float = 5000.0,
    trigger_mask_pixels: float = 5000.0,
    trigger_box_area_frac: float = 0.20,
) -> Dict[str, Any]:
    score = 0.0
    reasons = []
    role = _role(config)
    verifier_mode = _role_verifier_mode(config)

    box_score = _as_float(candidate.get("box_score", candidate.get("score")), 0.0)
    score += 2.0 * box_score
    reasons.append(f"box_score={box_score:.2f}")

    point_count = _as_int(candidate.get("object_point_count"), 0)
    min_points = _as_int(config.get("min_points"), 20)
    if point_count < min_points:
        score -= 3.0
        reasons.append(f"too_few_points={point_count}")
    else:
        point_bonus = min(0.55, math.log1p(point_count / max(float(min_points), 1.0)) * 0.16)
        score += point_bonus
        reasons.append(f"points={point_count}")

    max_points = config.get("max_points")
    if max_points is not None and point_count > _as_int(max_points, 0):
        score -= 2.0
        reasons.append(f"above_max_points={point_count}>{int(max_points)}")

    trigger_points = max(float(trigger_points), 1.0)
    if point_count > trigger_points:
        penalty = min(2.2, (point_count / trigger_points - 1.0) * 0.95)
        score -= penalty
        reasons.append(f"large_points_penalty={penalty:.2f}")

    mask_pixels = _as_int(candidate.get("mask_pixels"), 0)
    trigger_mask_pixels = max(float(trigger_mask_pixels), 1.0)
    if mask_pixels > trigger_mask_pixels:
        penalty = min(1.8, (mask_pixels / trigger_mask_pixels - 1.0) * 0.85)
        score -= penalty
        reasons.append(f"large_mask_penalty={penalty:.2f}")

    box_area_frac = candidate.get("box_area_frac")
    if box_area_frac is not None:
        box_area_frac = _as_float(box_area_frac)
        if box_area_frac > trigger_box_area_frac:
            penalty = min(1.4, (box_area_frac - trigger_box_area_frac) * 4.0)
            score -= penalty
            reasons.append(f"large_box_penalty={penalty:.2f}")
        elif box_area_frac < 0.001:
            score -= 0.25
            reasons.append(f"tiny_box_area={box_area_frac:.4f}")
        else:
            score += 0.15
            reasons.append(f"box_area={box_area_frac:.3f}")

    if _box_touches_edge(candidate.get("box"), candidate.get("image_shape")):
        score -= 0.45
        reasons.append("touches_image_edge")

    center = candidate.get("object_point_center")
    if center is None:
        score -= 0.8
        reasons.append("missing_center")
    else:
        center_z = _as_float(center[2])
        max_center_z = config.get("max_center_z")
        if max_center_z is not None and center_z > _as_float(max_center_z):
            score -= 2.2
            reasons.append(f"center_z_above_max={center_z:.3f}")
        soft_max_z = _workspace_soft_max_z(manifest)
        if soft_max_z is not None and center_z > soft_max_z:
            penalty = min(1.8, (center_z - soft_max_z) * 3.2)
            score -= penalty
            reasons.append(f"high_workspace_z_penalty={penalty:.2f}")
        else:
            score += 0.15
            reasons.append(f"center_z={center_z:.3f}")

    if candidate.get("label"):
        score += 0.08
        reasons.append("has_label")

    if role != "unknown":
        score = _apply_camera_role_prior(
            score,
            reasons,
            role,
            candidate,
            trigger_points=trigger_points,
            trigger_box_area_frac=trigger_box_area_frac,
        )
    score = _apply_distractor_penalty(score, reasons, role, candidate)

    return {
        "version": SCORER_VERSION,
        "role_category": role,
        "score": float(score),
        "reasons": reasons,
        "box_score": box_score,
        "point_count": point_count,
        "mask_pixels": mask_pixels,
    }


def score_phrase_summary_v2(
    *,
    query_name: str,
    phrase: str,
    phrase_rank: int,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    score = 0.0
    reasons = []
    role = _role(config)
    verifier_mode = _role_verifier_mode(config)
    fused_count = _as_int(summary.get("fused_point_count"), len(points) if points is not None else 0)
    if fused_count <= 0:
        return {
            "version": SCORER_VERSION,
            "query_name": query_name,
            "phrase": phrase,
            "phrase_rank": phrase_rank,
            "score": -100.0,
            "reasons": ["no_fused_points"],
            "fused_point_count": fused_count,
            "used_camera_count": 0,
        }

    used = [result for result in summary.get("camera_results", []) if result.get("used_for_fusion")]
    used_count = len(used)
    if used_count:
        score += 0.55 * used_count
        reasons.append(f"used_cameras={used_count}")
        mean_box_score = float(np.mean([_as_float(result.get("box_score")) for result in used]))
        score += 1.25 * mean_box_score
        reasons.append(f"mean_box_score={mean_box_score:.2f}")
        replay_scores = [
            _as_float((result.get("replay_score_info") or {}).get("score"))
            for result in used
            if result.get("replay_score_info") is not None
        ]
        if replay_scores:
            mean_replay_score = float(np.mean(replay_scores))
            replay_bonus = max(-0.8, min(0.8, 0.35 * mean_replay_score))
            score += replay_bonus
            reasons.append(f"mean_replay_score={mean_replay_score:.2f}")
    else:
        score -= 2.0
        reasons.append("no_used_cameras")

    min_points = _as_int(config.get("min_points"), 20)
    max_points = config.get("max_points")
    if fused_count >= min_points:
        point_bonus = min(0.5, math.log1p(fused_count / max(float(min_points), 1.0)) * 0.12)
        score += point_bonus
    else:
        score -= 1.5
        reasons.append(f"too_few_points={fused_count}")
    if max_points is not None:
        max_points_int = _as_int(max_points, 0)
        if fused_count > max_points_int:
            score -= 2.5
            reasons.append(f"too_many_points={fused_count}")
        elif fused_count > 0.8 * max_points_int:
            score -= 0.45
            reasons.append(f"large_point_count={fused_count}")

    centers = _finite_centers(used)
    mean_pairwise = _mean_pairwise_distance(centers)
    if mean_pairwise is not None:
        if mean_pairwise <= 0.06:
            score += 0.45
            reasons.append(f"center_consistent={mean_pairwise:.3f}")
        elif mean_pairwise <= 0.14:
            score -= 0.35
            reasons.append(f"center_spread={mean_pairwise:.3f}")
        else:
            penalty = min(1.8, (mean_pairwise - 0.14) * 6.0 + 0.55)
            score -= penalty
            reasons.append(f"center_inconsistent={mean_pairwise:.3f}")

    center = summary.get("fused_object_point_center")
    if center is not None:
        center_z = _as_float(center[2])
        if config.get("max_center_z") is not None and center_z > _as_float(config["max_center_z"]):
            score -= 2.2
            reasons.append(f"center_z_above_max={center_z:.3f}")
        manifest = {}
        soft_max_z = None
        manifest_path = summary.get("manifest")
        if manifest_path:
            try:
                import json
                import os

                if os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    soft_max_z = _workspace_soft_max_z(manifest)
            except Exception:
                soft_max_z = None
        if soft_max_z is not None and center_z > soft_max_z:
            penalty = min(1.6, (center_z - soft_max_z) * 3.0)
            score -= penalty
            reasons.append(f"high_center_z={center_z:.3f}>{soft_max_z:.3f}")
        else:
            score += 0.15
            reasons.append(f"center_z={center_z:.3f}")

    oversized = [
        result
        for result in used
        if _as_int(result.get("mask_pixels"), 0) > 7000
        or _as_int(result.get("object_point_count"), 0) > 7000
    ]
    if oversized:
        penalty = min(1.8, 0.55 * len(oversized))
        score -= penalty
        reasons.append(f"oversized_views={len(oversized)}")

    edge_views = [
        result
        for result in used
        if _box_touches_edge(result.get("box"), result.get("image_shape"))
    ]
    if edge_views:
        penalty = min(0.8, 0.25 * len(edge_views))
        score -= penalty
        reasons.append(f"edge_views={len(edge_views)}")

    phrase_rank_bonus = max(0.0, 0.18 - 0.05 * phrase_rank)
    score += phrase_rank_bonus
    reasons.append(f"rank_bonus={phrase_rank_bonus:.2f}")
    if role != "unknown":
        score = _apply_phrase_role_prior(
            score,
            reasons,
            role,
            fused_count,
            used,
            max_points,
        )
        if verifier_mode == "strict":
            score = _apply_strict_role_verifier(
                score,
                reasons,
                role,
                fused_count,
                used,
                centers,
            )
    return {
        "version": SCORER_VERSION,
        "role_category": role,
        "query_name": query_name,
        "phrase": phrase,
        "phrase_rank": phrase_rank,
        "score": float(score),
        "reasons": reasons,
        "fused_point_count": fused_count,
        "used_camera_count": used_count,
    }
