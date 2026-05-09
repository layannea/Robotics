import json
import os
from typing import Iterable, Optional

import numpy as np
from PIL import Image


DEFAULT_CAMERAS = ("front", "left_shoulder", "right_shoulder", "overhead", "wrist")


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    if np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def dump_current_obs(
    env,
    out_dir: str = "perception_dumps/latest",
    cameras: Optional[Iterable[str]] = None,
) -> str:
    """Dump RGB, point cloud, and oracle mask arrays from env.latest_obs.

    This intentionally keeps perception experiments outside the live RLBench
    control loop. The output can be consumed from a separate Python environment
    with GroundingDINO/SAM2 installed.
    """
    if env.latest_obs is None:
        raise ValueError("env.latest_obs is None. Call env.reset() first.")

    cameras = tuple(cameras or DEFAULT_CAMERAS)
    os.makedirs(out_dir, exist_ok=True)

    task_name = None
    if getattr(env, "task", None) is not None:
        try:
            task_name = env.task.get_name()
        except Exception:
            task_name = None

    object_name_to_ids = {}
    for name, ids in getattr(env, "name2ids", {}).items():
        object_name_to_ids[str(name)] = [int(obj_id) for obj_id in ids]

    manifest = {
        "task_name": task_name,
        "object_name_to_ids": object_name_to_ids,
        "cameras": [],
        "workspace_bounds_min": getattr(env, "workspace_bounds_min", None),
        "workspace_bounds_max": getattr(env, "workspace_bounds_max", None),
    }
    for key in ("workspace_bounds_min", "workspace_bounds_max"):
        if isinstance(manifest[key], np.ndarray):
            manifest[key] = manifest[key].tolist()

    for camera in cameras:
        rgb = _rgb_to_uint8(getattr(env.latest_obs, f"{camera}_rgb"))
        point_cloud = np.asarray(getattr(env.latest_obs, f"{camera}_point_cloud"))
        mask = np.asarray(getattr(env.latest_obs, f"{camera}_mask"))

        rgb_path = os.path.join(out_dir, f"{camera}_rgb.png")
        point_cloud_path = os.path.join(out_dir, f"{camera}_point_cloud.npy")
        mask_path = os.path.join(out_dir, f"{camera}_oracle_mask.npy")

        Image.fromarray(rgb).save(rgb_path)
        np.save(point_cloud_path, point_cloud)
        np.save(mask_path, mask)

        manifest["cameras"].append(
            {
                "name": camera,
                "rgb": rgb_path,
                "point_cloud": point_cloud_path,
                "oracle_mask": mask_path,
                "shape": list(rgb.shape[:2]),
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
