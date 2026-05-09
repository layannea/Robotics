import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = points.reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _center(points: np.ndarray) -> Optional[List[float]]:
    points = _finite_points(points)
    if len(points) == 0:
        return None
    return points.mean(axis=0).tolist()


def _manifest_object_names(manifest: Dict[str, Any]) -> List[str]:
    names = [str(name) for name in (manifest.get("object_name_to_ids") or {}).keys()]
    return sorted(names, key=len, reverse=True)


def _path_part_matches_object(part: str, object_name: str) -> bool:
    part = part.lower()
    variants = {object_name.lower(), object_name.lower().replace(" ", "_")}
    for variant in variants:
        if part == variant or part.startswith(f"{variant}_"):
            return True
        if part.startswith("perception_"):
            hint = part[len("perception_") :]
            if hint == variant or hint.startswith(f"{variant}_"):
                return True
    return False


def _infer_object_name(
    summary_path: str,
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    parts = summary_path.replace(os.sep, "/").split("/")
    for name in _manifest_object_names(manifest or {}):
        for part in reversed(parts):
            if _path_part_matches_object(part, name):
                return name

    for part in reversed(parts):
        if part.startswith("perception_") and part != "perception_dumps":
            object_hint = part[len("perception_") :]
            return object_hint
    text = str(summary.get("text") or "").strip().lower()
    for name in _manifest_object_names(manifest or {}):
        if name.lower() in text or name.lower().replace(" ", "_") in text:
            return name
    for name in ("rubbish", "bin", "button", "tomato"):
        if name in text:
            return name
    return None


def _infer_task_name(summary_path: str, manifest: Dict[str, Any]) -> Optional[str]:
    task_name = manifest.get("task_name")
    if task_name:
        return task_name

    parts = summary_path.replace(os.sep, "/").split("/")
    if "live" in parts:
        live_index = parts.index("live")
        if live_index + 1 < len(parts):
            return parts[live_index + 1]
    if "combined_env_smoke" in parts:
        return "combined_env_smoke"
    return None


def _load_oracle_map(value: Optional[str]) -> Dict[str, List[int]]:
    if not value:
        return {}
    if os.path.exists(value):
        data = _load_json(value)
    else:
        data = json.loads(value)
    return {str(k): [int(x) for x in v] for k, v in data.items()}


def _parse_oracle_id_args(values: Optional[Iterable[str]]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--oracle-id must look like object=1,2,3, got {value!r}")
        name, ids = value.split("=", 1)
        mapping[name.strip()] = [int(x.strip()) for x in ids.split(",") if x.strip()]
    return mapping


def _manifest_oracle_ids(
    manifest: Dict[str, Any],
    object_name: Optional[str],
    oracle_map: Dict[str, List[int]],
) -> Optional[List[int]]:
    if not object_name:
        return None
    if object_name in oracle_map:
        return oracle_map[object_name]
    lower_map = {key.lower(): value for key, value in oracle_map.items()}
    if object_name.lower() in lower_map:
        return lower_map[object_name.lower()]

    manifest_map = manifest.get("object_name_to_ids") or {}
    if object_name in manifest_map:
        return [int(x) for x in manifest_map[object_name]]
    lower_manifest_map = {key.lower(): value for key, value in manifest_map.items()}
    if object_name.lower() in lower_manifest_map:
        return [int(x) for x in lower_manifest_map[object_name.lower()]]
    return None


def _camera_by_name(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {camera["name"]: camera for camera in manifest.get("cameras", [])}


def _points_from_oracle_camera(camera: Dict[str, Any], oracle_ids: List[int]) -> np.ndarray:
    point_cloud = np.load(camera["point_cloud"])
    oracle_mask = np.load(camera["oracle_mask"])
    chosen = np.isin(oracle_mask, oracle_ids)
    if point_cloud.shape[:2] == chosen.shape:
        points = point_cloud[chosen]
    elif point_cloud.reshape(-1, 3).shape[0] == chosen.size:
        points = point_cloud.reshape(-1, 3)[chosen.reshape(-1)]
    else:
        raise ValueError(
            f"Point cloud shape {point_cloud.shape} is incompatible with oracle mask {chosen.shape}"
        )
    return _finite_points(points)


def _mask_iou(pred_mask_path: Optional[str], camera: Dict[str, Any], oracle_ids: List[int]) -> Optional[float]:
    if not pred_mask_path or not os.path.exists(pred_mask_path):
        return None
    pred_mask = np.load(pred_mask_path).astype(bool)
    oracle_mask = np.isin(np.load(camera["oracle_mask"]), oracle_ids)
    if pred_mask.shape != oracle_mask.shape:
        pred_mask = pred_mask.reshape(-1)
        oracle_mask = oracle_mask.reshape(-1)
    intersection = np.logical_and(pred_mask, oracle_mask).sum()
    union = np.logical_or(pred_mask, oracle_mask).sum()
    if union == 0:
        return None
    return float(intersection / union)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _summary_points(summary: Dict[str, Any]) -> np.ndarray:
    points_path = summary.get("fused_object_points")
    if points_path and os.path.exists(points_path):
        return _finite_points(np.load(points_path))
    used_points = []
    for result in summary.get("camera_results", []):
        if result.get("used_for_fusion") and result.get("object_points"):
            path = result["object_points"]
            if os.path.exists(path):
                used_points.append(_finite_points(np.load(path)))
    if used_points:
        return _finite_points(np.concatenate(used_points, axis=0))
    return np.empty((0, 3), dtype=np.float32)


def _oracle_points_from_manifest(manifest: Dict[str, Any], oracle_ids: List[int]) -> np.ndarray:
    oracle_points = []
    for camera in manifest.get("cameras", []):
        if "oracle_mask" not in camera or "point_cloud" not in camera:
            continue
        oracle_points.append(_points_from_oracle_camera(camera, oracle_ids))
    if not oracle_points:
        return np.empty((0, 3), dtype=np.float32)
    return _finite_points(np.concatenate(oracle_points, axis=0))


def _numeric_suffix(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"(\d+)$", value)
    if not match:
        return None
    return int(match.group(1))


def _instance_min_cameras(object_name: Optional[str]) -> int:
    if object_name and re.fullmatch(r"tomato\d*", object_name.lower()):
        return 2
    return 1


def _select_instance(summary: Dict[str, Any], object_name: Optional[str]) -> Optional[Dict[str, Any]]:
    instances = list(summary.get("instances") or [])
    if not instances:
        return None

    min_cameras = _instance_min_cameras(object_name)
    usable = [
        instance
        for instance in instances
        if len(instance.get("cameras") or []) >= min_cameras
        and instance.get("fused_object_point_center") is not None
    ]
    if not usable:
        usable = [instance for instance in instances if instance.get("fused_object_point_center") is not None]
    if not usable:
        return None

    suffix = _numeric_suffix(object_name)
    if suffix is not None:
        usable = sorted(usable, key=lambda inst: inst["fused_object_point_center"][1])
        return usable[min(max(suffix - 1, 0), len(usable) - 1)]
    return sorted(
        usable,
        key=lambda inst: (
            -int(inst.get("candidate_count") or 0),
            -int(inst.get("fused_point_count") or 0),
            int(inst.get("instance_id") or 0),
        ),
    )[0]


def _instance_points(instance: Optional[Dict[str, Any]]) -> np.ndarray:
    if not instance:
        return np.empty((0, 3), dtype=np.float32)
    points_path = instance.get("fused_object_points")
    if points_path and os.path.exists(points_path):
        return _finite_points(np.load(points_path))
    return np.empty((0, 3), dtype=np.float32)


def _candidate_lookup(summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup = {}
    for index, candidate in enumerate(summary.get("candidates") or []):
        lookup[index] = candidate
    return lookup


def _instance_mask_ious(
    summary: Dict[str, Any],
    instance: Optional[Dict[str, Any]],
    manifest: Dict[str, Any],
    oracle_ids: List[int],
) -> List[float]:
    if not instance:
        return []
    cameras = _camera_by_name(manifest)
    candidates = _candidate_lookup(summary)
    ious = []
    for candidate_index in instance.get("candidate_indices") or []:
        candidate = candidates.get(int(candidate_index))
        if not candidate:
            continue
        camera = cameras.get(candidate.get("camera"))
        if not camera or "oracle_mask" not in camera:
            continue
        iou = _mask_iou(candidate.get("mask"), camera, oracle_ids)
        if iou is not None:
            ious.append(iou)
    return ious


def evaluate_summary(
    summary_path: str,
    object_name: Optional[str],
    oracle_map: Dict[str, List[int]],
    center_pass_threshold: float,
    min_pred_points: int,
) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    manifest_path = summary.get("manifest")
    manifest = _load_json(manifest_path) if manifest_path and os.path.exists(manifest_path) else {}
    inferred_object = object_name or _infer_object_name(summary_path, summary, manifest)
    pred_points = _summary_points(summary)
    pred_center = _center(pred_points)

    camera_results = summary.get("camera_results", [])
    accepted = [result for result in camera_results if result.get("accepted")]
    used = [result for result in camera_results if result.get("used_for_fusion")]
    used_scores = [float(result.get("box_score", 0.0)) for result in used if result.get("box_score") is not None]
    accepted_scores = [
        float(result.get("box_score", 0.0))
        for result in accepted
        if result.get("box_score") is not None
    ]

    row: Dict[str, Any] = {
        "summary": summary_path,
        "manifest": manifest_path,
        "result_type": "summary",
        "task": _infer_task_name(summary_path, manifest),
        "object": inferred_object,
        "text": summary.get("text"),
        "instance_id": None,
        "instance_count": None,
        "pred_points": int(len(pred_points)),
        "pred_center": pred_center,
        "pred_center_z": pred_center[2] if pred_center else None,
        "accepted_cameras": len(accepted),
        "used_cameras": len(used),
        "mean_used_box_score": _mean(used_scores),
        "mean_accepted_box_score": _mean(accepted_scores),
        "oracle_available": False,
        "oracle_points": None,
        "oracle_center": None,
        "center_dist_m": None,
        "mean_mask_iou": None,
        "pass": int(len(pred_points) >= min_pred_points),
        "fail_reason": None,
    }

    oracle_ids = _manifest_oracle_ids(manifest, inferred_object, oracle_map)
    if oracle_ids:
        camera_lookup = _camera_by_name(manifest)
        oracle_points = []
        ious = []
        for result in camera_results:
            camera = camera_lookup.get(result.get("camera"))
            if not camera or "oracle_mask" not in camera:
                continue
            oracle_points.append(_points_from_oracle_camera(camera, oracle_ids))
            iou = _mask_iou(result.get("mask"), camera, oracle_ids)
            if iou is not None:
                ious.append(iou)
        oracle_points_arr = (
            _finite_points(np.concatenate(oracle_points, axis=0))
            if oracle_points
            else np.empty((0, 3), dtype=np.float32)
        )
        oracle_center = _center(oracle_points_arr)
        center_dist = None
        if pred_center is not None and oracle_center is not None:
            center_dist = float(np.linalg.norm(np.asarray(pred_center) - np.asarray(oracle_center)))

        row.update(
            {
                "oracle_available": True,
                "oracle_ids": oracle_ids,
                "oracle_points": int(len(oracle_points_arr)),
                "oracle_center": oracle_center,
                "center_dist_m": center_dist,
                "mean_mask_iou": _mean(ious),
            }
        )
        row["pass"] = bool(
            len(pred_points) >= min_pred_points
            and center_dist is not None
            and center_dist <= center_pass_threshold
        )

    if not row["pass"]:
        if len(pred_points) < min_pred_points:
            row["fail_reason"] = "too_few_pred_points"
        elif row["oracle_available"] and row["center_dist_m"] is not None:
            row["fail_reason"] = "center_too_far"
        elif row["oracle_available"]:
            row["fail_reason"] = "missing_center"
    return row


def evaluate_instances(
    instances_path: str,
    object_name: Optional[str],
    oracle_map: Dict[str, List[int]],
    center_pass_threshold: float,
    min_pred_points: int,
) -> Dict[str, Any]:
    summary = _load_json(instances_path)
    manifest_path = summary.get("manifest")
    manifest = _load_json(manifest_path) if manifest_path and os.path.exists(manifest_path) else {}
    inferred_object = object_name or _infer_object_name(instances_path, summary, manifest)
    instance = _select_instance(summary, inferred_object)
    pred_points = _instance_points(instance)
    pred_center = _center(pred_points)
    members = instance.get("members") if instance else []
    used_scores = [float(member.get("box_score", 0.0)) for member in members if member.get("box_score") is not None]

    row: Dict[str, Any] = {
        "summary": instances_path,
        "manifest": manifest_path,
        "result_type": "instances",
        "task": _infer_task_name(instances_path, manifest),
        "object": inferred_object,
        "text": summary.get("text"),
        "instance_id": instance.get("instance_id") if instance else None,
        "instance_count": summary.get("instance_count"),
        "pred_points": int(len(pred_points)),
        "pred_center": pred_center,
        "pred_center_z": pred_center[2] if pred_center else None,
        "accepted_cameras": int(summary.get("accepted_candidate_count") or 0),
        "used_cameras": len(instance.get("cameras") or []) if instance else 0,
        "mean_used_box_score": _mean(used_scores),
        "mean_accepted_box_score": _mean(used_scores),
        "oracle_available": False,
        "oracle_points": None,
        "oracle_center": None,
        "center_dist_m": None,
        "mean_mask_iou": None,
        "pass": int(len(pred_points) >= min_pred_points),
        "fail_reason": None,
    }

    oracle_ids = _manifest_oracle_ids(manifest, inferred_object, oracle_map)
    if oracle_ids:
        oracle_points = _oracle_points_from_manifest(manifest, oracle_ids)
        oracle_center = _center(oracle_points)
        center_dist = None
        if pred_center is not None and oracle_center is not None:
            center_dist = float(np.linalg.norm(np.asarray(pred_center) - np.asarray(oracle_center)))
        row.update(
            {
                "oracle_available": True,
                "oracle_ids": oracle_ids,
                "oracle_points": int(len(oracle_points)),
                "oracle_center": oracle_center,
                "center_dist_m": center_dist,
                "mean_mask_iou": _mean(_instance_mask_ious(summary, instance, manifest, oracle_ids)),
            }
        )
        row["pass"] = bool(
            len(pred_points) >= min_pred_points
            and center_dist is not None
            and center_dist <= center_pass_threshold
        )

    if not row["pass"]:
        if instance is None:
            row["fail_reason"] = "missing_instance"
        elif len(pred_points) < min_pred_points:
            row["fail_reason"] = "too_few_pred_points"
        elif row["oracle_available"] and row["center_dist_m"] is not None:
            row["fail_reason"] = "center_too_far"
        elif row["oracle_available"]:
            row["fail_reason"] = "missing_center"
    return row


def evaluate_result(
    result_path: str,
    object_name: Optional[str],
    oracle_map: Dict[str, List[int]],
    center_pass_threshold: float,
    min_pred_points: int,
) -> Dict[str, Any]:
    if os.path.basename(result_path) == "instances.json":
        return evaluate_instances(
            instances_path=result_path,
            object_name=object_name,
            oracle_map=oracle_map,
            center_pass_threshold=center_pass_threshold,
            min_pred_points=min_pred_points,
        )
    return evaluate_summary(
        summary_path=result_path,
        object_name=object_name,
        oracle_map=oracle_map,
        center_pass_threshold=center_pass_threshold,
        min_pred_points=min_pred_points,
    )


def _resolve_summary_paths(args) -> List[str]:
    paths = []
    for pattern in args.summaries_glob or []:
        paths.extend(glob.glob(pattern, recursive=True))
    paths.extend(args.summary or [])
    deduped = []
    for path in paths:
        path = os.path.abspath(path)
        if path not in deduped:
            deduped.append(path)
    return deduped


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "pass",
        "result_type",
        "task",
        "object",
        "text",
        "instance_id",
        "instance_count",
        "pred_points",
        "used_cameras",
        "accepted_cameras",
        "pred_center_z",
        "oracle_available",
        "oracle_points",
        "center_dist_m",
        "mean_mask_iou",
        "mean_used_box_score",
        "fail_reason",
        "summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GroundingDINO/SAM2 perception summaries against RLBench oracle masks when available."
    )
    parser.add_argument("--summary", action="append", help="Path to a summary.json or instances.json result.")
    parser.add_argument(
        "--summaries-glob",
        action="append",
        help="Glob for summary.json or instances.json files. Quote the pattern in the shell.",
    )
    parser.add_argument("--object", dest="object_name", help="Object name override for all summaries.")
    parser.add_argument(
        "--oracle-map",
        help='JSON string or path, e.g. {"rubbish":[12,13],"bin":[20]}. Manifest object_name_to_ids is used first when present.',
    )
    parser.add_argument(
        "--oracle-id",
        action="append",
        help="Manual oracle id mapping, e.g. --oracle-id rubbish=12,13. Can be repeated.",
    )
    parser.add_argument("--center-pass-threshold", type=float, default=0.08)
    parser.add_argument("--min-pred-points", type=int, default=20)
    parser.add_argument("--out-json")
    parser.add_argument("--out-csv")
    args = parser.parse_args()

    summary_paths = _resolve_summary_paths(args)
    if not summary_paths:
        raise ValueError("Pass at least one --summary or --summaries-glob.")

    oracle_map = _load_oracle_map(args.oracle_map)
    oracle_map.update(_parse_oracle_id_args(args.oracle_id))

    rows = [
        evaluate_result(
            result_path=path,
            object_name=args.object_name,
            oracle_map=oracle_map,
            center_pass_threshold=args.center_pass_threshold,
            min_pred_points=args.min_pred_points,
        )
        for path in summary_paths
    ]
    rows.sort(key=lambda row: (str(row.get("task")), str(row.get("object")), str(row.get("text"))))

    result = {
        "summary_count": len(rows),
        "pass_count": int(sum(bool(row["pass"]) for row in rows)),
        "oracle_count": int(sum(bool(row["oracle_available"]) for row in rows)),
        "rows": rows,
    }

    if args.out_json:
        _save_json(args.out_json, result)
    if args.out_csv:
        _write_csv(args.out_csv, rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
