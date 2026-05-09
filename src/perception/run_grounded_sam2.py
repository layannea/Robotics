import argparse
import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def _to_device(inputs: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _post_process_grounding(processor, outputs, inputs, image_size, box_threshold, text_threshold):
    target_sizes = torch.tensor([image_size[::-1]], device=outputs.logits.device)
    if hasattr(processor, "post_process_grounded_object_detection"):
        try:
            return processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            try:
                return processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.get("input_ids"),
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                return processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
    return processor.post_process_object_detection(
        outputs,
        threshold=box_threshold,
        target_sizes=target_sizes,
    )[0]


def _as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _result_labels(result: Dict[str, Any]) -> List[str]:
    if "text_labels" in result:
        return list(result["text_labels"])
    if "labels" in result:
        labels = result["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        return [str(label) for label in labels]
    return []


def _draw_overlay(image_rgb: np.ndarray, box: np.ndarray, mask: np.ndarray, score: float, label: str) -> np.ndarray:
    overlay = image_rgb.copy()
    color = np.array([0, 255, 120], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

    x0, y0, x1, y1 = box.astype(int)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 80, 20), 2)
    caption = f"{label} {score:.2f}".strip()
    cv2.putText(
        overlay,
        caption,
        (max(x0, 0), max(y0 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 80, 20),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _clamp_box(box: np.ndarray, image_shape) -> np.ndarray:
    height, width = image_shape[:2]
    clamped = box.astype(np.float32).copy()
    clamped[[0, 2]] = np.clip(clamped[[0, 2]], 0, width - 1)
    clamped[[1, 3]] = np.clip(clamped[[1, 3]], 0, height - 1)
    return clamped


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
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def main():
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM2 on one RLBench RGB image and map the mask to point cloud."
    )
    parser.add_argument("--rgb", required=True, help="Path to an RGB image, e.g. front_rgb.png.")
    parser.add_argument("--text", required=True, help="Text prompt, e.g. 'button.' or 'purple base button.'.")
    parser.add_argument("--point-cloud", help="Optional matching RLBench point cloud .npy for this camera.")
    parser.add_argument("--out-dir", default="perception_dumps/grounded_sam2")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--box-index", type=int, help="Use a specific GroundingDINO candidate instead of the best score.")
    parser.add_argument(
        "--grounding-scale",
        type=float,
        default=1.0,
        help="Upscale the RGB image for GroundingDINO only, then map boxes back to original pixels.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image = Image.open(args.rgb).convert("RGB")
    image_rgb = np.array(image)
    if args.grounding_scale <= 0:
        raise ValueError("--grounding-scale must be positive")
    if args.grounding_scale == 1.0:
        grounding_image = image
    else:
        grounding_size = (
            int(round(image.width * args.grounding_scale)),
            int(round(image.height * args.grounding_scale)),
        )
        grounding_image = image.resize(grounding_size, Image.Resampling.BICUBIC)
    text = args.text if args.text.strip().endswith(".") else f"{args.text.strip()}."

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model)
    grounding_model = grounding_model.to(args.device).eval()

    inputs = processor(images=grounding_image, text=text, return_tensors="pt")
    inputs = _to_device(inputs, args.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    result = _post_process_grounding(
        processor,
        outputs,
        inputs,
        grounding_image.size,
        args.box_threshold,
        args.text_threshold,
    )

    boxes = result.get("boxes")
    scores = result.get("scores")
    if boxes is None or len(boxes) == 0:
        raise RuntimeError(f"No boxes found for prompt: {args.text!r}")

    boxes_np = np.asarray(_as_list(boxes), dtype=np.float32)
    boxes_np = boxes_np / args.grounding_scale
    scores_np = np.asarray(_as_list(scores), dtype=np.float32)
    labels = _result_labels(result)
    if args.box_index is None:
        best_idx = int(np.argmax(scores_np))
    else:
        if args.box_index < 0 or args.box_index >= len(boxes_np):
            raise ValueError(f"--box-index must be between 0 and {len(boxes_np) - 1}")
        best_idx = args.box_index
    best_box = _clamp_box(boxes_np[best_idx], image_rgb.shape)
    best_score = float(scores_np[best_idx])
    best_label = labels[best_idx] if best_idx < len(labels) else args.text

    sam2_predictor = SAM2ImagePredictor.from_pretrained(args.sam2_model, device=args.device)
    sam2_predictor.set_image(image_rgb)
    masks, mask_scores, _ = sam2_predictor.predict(
        box=best_box,
        multimask_output=True,
    )
    mask_scores = np.asarray(mask_scores)
    best_mask = masks[int(mask_scores.argmax())].astype(bool)

    mask_path = os.path.join(args.out_dir, "mask.npy")
    overlay_path = os.path.join(args.out_dir, "overlay.png")
    result_path = os.path.join(args.out_dir, "result.json")
    np.save(mask_path, best_mask)

    overlay = _draw_overlay(image_rgb, best_box, best_mask, best_score, best_label)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    output = {
        "text": args.text,
        "grounding_model": args.grounding_model,
        "sam2_model": args.sam2_model,
        "grounding_scale": args.grounding_scale,
        "selected_box_index": best_idx,
        "box": best_box.tolist(),
        "box_score": best_score,
        "label": best_label,
        "candidates": [
            {
                "index": idx,
                "box": _clamp_box(box, image_rgb.shape).tolist(),
                "score": float(scores_np[idx]),
                "label": labels[idx] if idx < len(labels) else "",
            }
            for idx, box in enumerate(boxes_np)
        ],
        "mask_pixels": int(best_mask.sum()),
        "mask": mask_path,
        "overlay": overlay_path,
    }

    if args.point_cloud:
        object_points = _points_from_mask(args.point_cloud, best_mask)
        object_points_path = os.path.join(args.out_dir, "object_points.npy")
        np.save(object_points_path, object_points)
        output["point_cloud"] = args.point_cloud
        output["object_points"] = object_points_path
        output["object_point_count"] = int(len(object_points))
        if len(object_points) > 0:
            output["object_point_center"] = object_points.mean(axis=0).tolist()

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
