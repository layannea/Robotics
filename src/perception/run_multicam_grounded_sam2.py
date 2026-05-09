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


def _choose_candidate(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    image_shape,
    box_index: Optional[int],
    selection: str,
) -> int:
    if box_index is not None:
        if box_index < 0 or box_index >= len(boxes):
            raise ValueError(f"--box-index must be between 0 and {len(boxes) - 1}")
        return box_index

    # Prefer candidates that have a text label when available, then use score.
    labeled_indices = [idx for idx, label in enumerate(labels) if label]
    if labeled_indices and selection == "smallest_labeled_area":
        return min(
            labeled_indices,
            key=lambda idx: max(0.0, boxes[idx][2] - boxes[idx][0])
            * max(0.0, boxes[idx][3] - boxes[idx][1]),
        )
    if labeled_indices:
        return max(labeled_indices, key=lambda idx: scores[idx])
    return int(np.argmax(scores))


def _run_one_camera(
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
    box_index: Optional[int],
    selection: str,
) -> Dict[str, Any]:
    image = Image.open(camera["rgb"]).convert("RGB")
    image_rgb = np.array(image)
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
        return {
            "camera": camera["name"],
            "accepted": False,
            "reason": "no_boxes",
            "rgb": camera["rgb"],
        }

    boxes_np = np.asarray(_as_list(boxes), dtype=np.float32) / grounding_scale
    scores_np = np.asarray(_as_list(scores), dtype=np.float32)
    labels = _result_labels(result)
    selected_idx = _choose_candidate(boxes_np, scores_np, labels, image_rgb.shape, box_index, selection)
    selected_box = _clamp_box(boxes_np[selected_idx], image_rgb.shape)
    selected_score = float(scores_np[selected_idx])
    selected_label = labels[selected_idx] if selected_idx < len(labels) else ""

    sam2_predictor.set_image(image_rgb)
    masks, mask_scores, _ = sam2_predictor.predict(box=selected_box, multimask_output=True)
    mask_idx = int(np.asarray(mask_scores).argmax())
    mask = masks[mask_idx].astype(bool)
    object_points = _points_from_mask(camera["point_cloud"], mask)

    camera_out_dir = os.path.join(out_dir, camera["name"])
    os.makedirs(camera_out_dir, exist_ok=True)
    mask_path = os.path.join(camera_out_dir, "mask.npy")
    points_path = os.path.join(camera_out_dir, "object_points.npy")
    overlay_path = os.path.join(camera_out_dir, "overlay.png")
    np.save(mask_path, mask)
    np.save(points_path, object_points)

    overlay = _draw_overlay(image_rgb, selected_box, mask, selected_score, selected_label or text)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    center = object_points.mean(axis=0).tolist() if len(object_points) else None
    return {
        "camera": camera["name"],
        "accepted": True,
        "rgb": camera["rgb"],
        "point_cloud": camera["point_cloud"],
        "selected_box_index": selected_idx,
        "box": selected_box.tolist(),
        "box_score": selected_score,
        "label": selected_label,
        "mask_pixels": int(mask.sum()),
        "object_point_count": int(len(object_points)),
        "object_point_center": center,
        "mask": mask_path,
        "object_points": points_path,
        "overlay": overlay_path,
        "candidates": [
            {
                "index": idx,
                "box": _clamp_box(box, image_rgb.shape).tolist(),
                "score": float(scores_np[idx]),
                "label": labels[idx] if idx < len(labels) else "",
            }
            for idx, box in enumerate(boxes_np)
        ],
    }


def _passes_filters(
    result: Dict[str, Any],
    min_points: int,
    min_score: float,
    max_center_z: Optional[float],
    max_points: Optional[int],
) -> bool:
    if not result.get("accepted"):
        result["used_for_fusion"] = False
        return False
    if result["object_point_count"] < min_points:
        result["used_for_fusion"] = False
        result["reject_reason"] = "too_few_points"
        return False
    if max_points is not None and result["object_point_count"] > max_points:
        result["used_for_fusion"] = False
        result["reject_reason"] = "too_many_points"
        return False
    if result["box_score"] < min_score:
        result["used_for_fusion"] = False
        result["reject_reason"] = "low_score"
        return False
    center = result.get("object_point_center")
    if max_center_z is not None and center is not None and center[2] > max_center_z:
        result["used_for_fusion"] = False
        result["reject_reason"] = "center_z_too_high"
        return False
    result["used_for_fusion"] = True
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM2 on multiple RLBench cameras and fuse their 3D object points."
    )
    parser.add_argument("--manifest", required=True, help="Path to dump_current_obs manifest.json.")
    parser.add_argument("--text", required=True, help="Text prompt, e.g. 'red button'.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cameras", nargs="*", help="Camera names to use. Defaults to all cameras in manifest.")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-scale", type=float, default=4.0)
    parser.add_argument("--box-index", type=int)
    parser.add_argument(
        "--selection",
        choices=("best_score", "smallest_labeled_area"),
        default="best_score",
        help="How to choose a GroundingDINO candidate when --box-index is not set.",
    )
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--max-center-z", type=float)
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

    results = []
    fused_points = []
    for camera in cameras:
        result = _run_one_camera(
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
            box_index=args.box_index,
            selection=args.selection,
        )
        if _passes_filters(result, args.min_points, args.min_score, args.max_center_z, args.max_points):
            fused_points.append(np.load(result["object_points"]))
        results.append(result)

    if fused_points:
        fused_points_arr = np.concatenate(fused_points, axis=0)
    else:
        fused_points_arr = np.empty((0, 3), dtype=np.float32)

    fused_points_path = os.path.join(args.out_dir, "fused_object_points.npy")
    np.save(fused_points_path, fused_points_arr)

    summary = {
        "text": args.text,
        "manifest": args.manifest,
        "grounding_model": args.grounding_model,
        "sam2_model": args.sam2_model,
        "grounding_scale": args.grounding_scale,
        "box_threshold": args.box_threshold,
        "selection": args.selection,
        "min_points": args.min_points,
        "max_points": args.max_points,
        "min_score": args.min_score,
        "max_center_z": args.max_center_z,
        "camera_results": results,
        "fused_object_points": fused_points_path,
        "fused_point_count": int(len(fused_points_arr)),
        "fused_object_point_center": fused_points_arr.mean(axis=0).tolist()
        if len(fused_points_arr)
        else None,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
