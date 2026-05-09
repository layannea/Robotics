import argparse
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from .run_grounded_sam2 import (
        _as_list,
        _clamp_box,
        _draw_overlay,
        _post_process_grounding,
        _result_labels,
        _to_device,
    )
except ImportError:
    from run_grounded_sam2 import (
        _as_list,
        _clamp_box,
        _draw_overlay,
        _post_process_grounding,
        _result_labels,
        _to_device,
    )


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _camera_entries(manifest: Dict[str, Any], names: Optional[List[str]]) -> List[Dict[str, Any]]:
    entries = manifest["cameras"]
    if not names:
        return entries
    requested = set(names)
    return [entry for entry in entries if entry["name"] in requested]


def _scale_image(image: Image.Image, scale: float) -> Image.Image:
    if scale <= 0:
        raise ValueError("--grounding-scale must be positive")
    if scale == 1.0:
        return image
    size = (int(round(image.width * scale)), int(round(image.height * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


def _points_from_mask(point_cloud_path: str, mask: np.ndarray) -> np.ndarray:
    point_cloud = np.load(point_cloud_path)
    if point_cloud.shape[:2] == mask.shape:
        points = point_cloud[mask]
    elif point_cloud.reshape(-1, 3).shape[0] == mask.size:
        points = point_cloud.reshape(-1, 3)[mask.reshape(-1)]
    else:
        raise ValueError(
            f"Point cloud shape {point_cloud.shape} is incompatible with mask shape {mask.shape}"
        )
    return points[np.isfinite(points).all(axis=1)]


def _box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _passes_candidate_filters(
    candidate: Dict[str, Any],
    min_points: int,
    max_points: Optional[int],
    min_score: float,
    max_center_z: Optional[float],
    max_box_area_frac: Optional[float],
) -> bool:
    if candidate["box_score"] < min_score:
        candidate["accepted"] = False
        candidate["reject_reason"] = "low_score"
        return False
    if candidate["object_point_count"] < min_points:
        candidate["accepted"] = False
        candidate["reject_reason"] = "too_few_points"
        return False
    if max_points is not None and candidate["object_point_count"] > max_points:
        candidate["accepted"] = False
        candidate["reject_reason"] = "too_many_points"
        return False
    center = candidate.get("object_point_center")
    if max_center_z is not None and center is not None and center[2] > max_center_z:
        candidate["accepted"] = False
        candidate["reject_reason"] = "center_z_too_high"
        return False
    if max_box_area_frac is not None and candidate["box_area_frac"] > max_box_area_frac:
        candidate["accepted"] = False
        candidate["reject_reason"] = "box_too_large"
        return False
    candidate["accepted"] = True
    return True


def _run_camera_candidates(
    camera: Dict[str, Any],
    text: str,
    processor,
    grounding_model,
    sam2_predictor,
    device: str,
    out_dir: str,
    grounding_scale: float,
    box_threshold: float,
    text_threshold: float,
    labeled_only: bool,
    max_candidates_per_camera: Optional[int],
    min_points: int,
    max_points: Optional[int],
    min_score: float,
    max_center_z: Optional[float],
    max_box_area_frac: Optional[float],
) -> List[Dict[str, Any]]:
    image = Image.open(camera["rgb"]).convert("RGB")
    image_rgb = np.array(image)
    image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    grounding_image = _scale_image(image, grounding_scale)

    inputs = processor(images=grounding_image, text=text, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    result = _post_process_grounding(
        processor,
        outputs,
        inputs,
        grounding_image.size,
        box_threshold,
        text_threshold,
    )
    boxes = result.get("boxes")
    scores = result.get("scores")
    if boxes is None or len(boxes) == 0:
        return []

    boxes_np = np.asarray(_as_list(boxes), dtype=np.float32) / grounding_scale
    scores_np = np.asarray(_as_list(scores), dtype=np.float32)
    labels = _result_labels(result)

    candidate_indices = list(range(len(boxes_np)))
    if labeled_only:
        candidate_indices = [idx for idx in candidate_indices if idx < len(labels) and labels[idx]]
    candidate_indices.sort(key=lambda idx: scores_np[idx], reverse=True)
    if max_candidates_per_camera is not None:
        candidate_indices = candidate_indices[:max_candidates_per_camera]

    camera_out_dir = os.path.join(out_dir, camera["name"])
    os.makedirs(camera_out_dir, exist_ok=True)
    sam2_predictor.set_image(image_rgb)

    candidates = []
    overlays = []
    for rank, idx in enumerate(candidate_indices):
        box = _clamp_box(boxes_np[idx], image_rgb.shape)
        label = labels[idx] if idx < len(labels) else ""
        score = float(scores_np[idx])
        masks, mask_scores, _ = sam2_predictor.predict(box=box, multimask_output=True)
        mask = masks[int(np.asarray(mask_scores).argmax())].astype(bool)
        object_points = _points_from_mask(camera["point_cloud"], mask)

        candidate_name = f"candidate_{rank:02d}_box_{idx:02d}"
        mask_path = os.path.join(camera_out_dir, f"{candidate_name}_mask.npy")
        points_path = os.path.join(camera_out_dir, f"{candidate_name}_points.npy")
        overlay_path = os.path.join(camera_out_dir, f"{candidate_name}_overlay.png")
        np.save(mask_path, mask)
        np.save(points_path, object_points)
        overlay = _draw_overlay(image_rgb, box, mask, score, label or text)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlays.append(overlay_path)

        center = object_points.mean(axis=0).tolist() if len(object_points) else None
        candidate = {
            "camera": camera["name"],
            "rgb": camera["rgb"],
            "point_cloud": camera["point_cloud"],
            "image_shape": list(image_rgb.shape[:2]),
            "source_box_index": int(idx),
            "rank": int(rank),
            "box": box.tolist(),
            "box_area_frac": _box_area(box) / image_area,
            "box_score": score,
            "label": label,
            "mask_pixels": int(mask.sum()),
            "object_point_count": int(len(object_points)),
            "object_point_center": center,
            "mask": mask_path,
            "object_points": points_path,
            "overlay": overlay_path,
        }
        _passes_candidate_filters(
            candidate,
            min_points=min_points,
            max_points=max_points,
            min_score=min_score,
            max_center_z=max_center_z,
            max_box_area_frac=max_box_area_frac,
        )
        candidates.append(candidate)

    return candidates


def _cluster_candidates(candidates: List[Dict[str, Any]], distance_threshold: float) -> List[List[int]]:
    accepted = [
        (idx, np.asarray(candidate["object_point_center"], dtype=np.float32))
        for idx, candidate in enumerate(candidates)
        if candidate.get("accepted") and candidate.get("object_point_center") is not None
    ]
    clusters: List[List[int]] = []
    for idx, center in accepted:
        best_cluster = None
        best_dist = None
        for cluster_idx, cluster in enumerate(clusters):
            cluster_centers = [
                np.asarray(candidates[candidate_idx]["object_point_center"], dtype=np.float32)
                for candidate_idx in cluster
            ]
            dist = float(np.min([np.linalg.norm(center - c) for c in cluster_centers]))
            if dist <= distance_threshold and (best_dist is None or dist < best_dist):
                best_cluster = cluster_idx
                best_dist = dist
        if best_cluster is None:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)
    return clusters


def _save_instances(
    candidates: List[Dict[str, Any]],
    clusters: List[List[int]],
    out_dir: str,
) -> List[Dict[str, Any]]:
    instance_records = []
    for cluster in clusters:
        points = [np.load(candidates[candidate_idx]["object_points"]) for candidate_idx in cluster]
        fused_points = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)
        used_candidates = [candidates[candidate_idx] for candidate_idx in cluster]
        instance_records.append(
            {
                "points": fused_points,
                "fused_point_count": int(len(fused_points)),
                "fused_object_point_center": fused_points.mean(axis=0).tolist()
                if len(fused_points)
                else None,
                "candidate_indices": cluster,
                "candidate_count": len(cluster),
                "cameras": sorted({candidate["camera"] for candidate in used_candidates}),
                "members": [
                    {
                        "camera": candidate["camera"],
                        "source_box_index": candidate["source_box_index"],
                        "box_score": candidate["box_score"],
                        "object_point_count": candidate["object_point_count"],
                        "object_point_center": candidate["object_point_center"],
                        "overlay": candidate["overlay"],
                    }
                    for candidate in used_candidates
                ],
            }
        )
    instance_records.sort(
        key=lambda instance: (
            -instance["candidate_count"],
            -instance["fused_point_count"],
            instance["fused_object_point_center"][0] if instance["fused_object_point_center"] else 0.0,
        )
    )
    instances = []
    for instance_id, instance in enumerate(instance_records):
        points_path = os.path.join(out_dir, f"instance_{instance_id:02d}_points.npy")
        np.save(points_path, instance.pop("points"))
        instance["instance_id"] = instance_id
        instance["fused_object_points"] = points_path
        instances.append(instance)
    return instances


def main():
    parser = argparse.ArgumentParser(
        description="Ground all visible candidates and cluster their 3D points into object instances."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cameras", nargs="*")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-scale", type=float, default=4.0)
    parser.add_argument("--max-candidates-per-camera", type=int, default=5)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--max-center-z", type=float)
    parser.add_argument("--max-box-area-frac", type=float, default=0.20)
    parser.add_argument("--cluster-distance", type=float, default=0.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    text = args.text if args.text.strip().endswith(".") else f"{args.text.strip()}."
    manifest = _load_manifest(args.manifest)
    cameras = _camera_entries(manifest, args.cameras)
    if not cameras:
        raise ValueError("No cameras selected.")

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model)
    grounding_model = grounding_model.to(args.device).eval()
    sam2_predictor = SAM2ImagePredictor.from_pretrained(args.sam2_model, device=args.device)

    candidates = []
    for camera in cameras:
        candidates.extend(
            _run_camera_candidates(
                camera=camera,
                text=text,
                processor=processor,
                grounding_model=grounding_model,
                sam2_predictor=sam2_predictor,
                device=args.device,
                out_dir=args.out_dir,
                grounding_scale=args.grounding_scale,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                labeled_only=not args.include_unlabeled,
                max_candidates_per_camera=args.max_candidates_per_camera,
                min_points=args.min_points,
                max_points=args.max_points,
                min_score=args.min_score,
                max_center_z=args.max_center_z,
                max_box_area_frac=args.max_box_area_frac,
            )
        )

    clusters = _cluster_candidates(candidates, args.cluster_distance)
    instances = _save_instances(candidates, clusters, args.out_dir)
    summary = {
        "text": args.text,
        "manifest": args.manifest,
        "grounding_model": args.grounding_model,
        "sam2_model": args.sam2_model,
        "grounding_scale": args.grounding_scale,
        "box_threshold": args.box_threshold,
        "max_candidates_per_camera": args.max_candidates_per_camera,
        "include_unlabeled": args.include_unlabeled,
        "min_points": args.min_points,
        "max_points": args.max_points,
        "min_score": args.min_score,
        "max_center_z": args.max_center_z,
        "max_box_area_frac": args.max_box_area_frac,
        "cluster_distance": args.cluster_distance,
        "candidate_count": len(candidates),
        "accepted_candidate_count": sum(1 for candidate in candidates if candidate.get("accepted")),
        "candidates": candidates,
        "instance_count": len(instances),
        "instances": instances,
    }
    summary_path = os.path.join(args.out_dir, "instances.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
