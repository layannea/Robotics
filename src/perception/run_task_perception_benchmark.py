import argparse
import json
import os
import re
import socket
import sys
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import rlbench.tasks as tasks
from envs.rlbench_env import VoxPoserRLBench

try:
    from .benchmark_perception import run_benchmark
except ImportError:
    from benchmark_perception import run_benchmark


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _snake_to_camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _task_mapping_path() -> str:
    return os.path.join(SRC_DIR, "envs", "task_object_names.json")


def _load_task_object_mapping() -> Dict[str, List[List[str]]]:
    return _load_json(_task_mapping_path())


def _task_class(task_name: str):
    class_name = _snake_to_camel(task_name)
    if not hasattr(tasks, class_name):
        raise ValueError(f"RLBench task class {class_name!r} not found for task {task_name!r}")
    return getattr(tasks, class_name)


def _default_out_root() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(SRC_DIR, "perception_dumps", "benchmark_live", stamp)


def _center(points: np.ndarray) -> Optional[List[float]]:
    points = np.asarray(points).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    return points.mean(axis=0).tolist()


def _benchmark_args(out_root: str, out_dir: str, args):
    class Args:
        pass

    bench_args = Args()
    bench_args.root = [out_root]
    bench_args.summaries_glob = None
    bench_args.summary = None
    bench_args.object_name = None
    bench_args.oracle_map = None
    bench_args.oracle_id = None
    bench_args.center_pass_threshold = args.center_pass_threshold
    bench_args.min_pred_points = args.min_pred_points
    bench_args.require_oracle = False
    bench_args.group_by = args.group_by
    bench_args.max_summaries = None
    bench_args.keep_going = True
    bench_args.selected_only = True
    bench_args.out_dir = out_dir
    bench_args.no_write = False
    return bench_args


def _write_benchmark_outputs(result: Dict[str, Any], out_dir: str):
    from benchmark_perception import _write_aggregate_csv
    from evaluate_perception import _write_csv

    os.makedirs(out_dir, exist_ok=True)
    _save_json(os.path.join(out_dir, "benchmark.json"), result)
    _write_csv(os.path.join(out_dir, "details.csv"), result["rows"])
    _write_aggregate_csv(os.path.join(out_dir, "aggregate.csv"), result["aggregate_rows"])


def _empty_benchmark_result() -> Dict[str, Any]:
    return {
        "summary_count": 0,
        "pass_count": 0,
        "pass_rate": None,
        "oracle_count": 0,
        "oracle_pass_count": 0,
        "oracle_pass_rate": None,
        "mean_center_dist_m": None,
        "mean_mask_iou": None,
        "rows": [],
        "aggregate_rows": [],
        "errors": [],
    }


def _check_tcp_endpoint(base_url: Optional[str], timeout: float = 2.0):
    if not base_url:
        return
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return
    except OSError as exc:
        raise ValueError(
            f"Cannot connect to VLM base URL {base_url!r} "
            f"at {parsed.hostname}:{port}. Start the local server or pass the correct --vlm-base-url."
        ) from exc


def _object_list(task_name: str, mapping: Dict[str, List[List[str]]], requested: Optional[List[str]]) -> List[str]:
    available = [item[0] for item in mapping[task_name]]
    if requested is None:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            f"Task {task_name!r} does not expose objects {missing}; available: {available}"
        )
    return requested


def run_live_benchmark(args) -> Dict[str, Any]:
    mapping = _load_task_object_mapping()
    task_names = _parse_csv(args.tasks) or list(mapping.keys())
    unknown_tasks = [task_name for task_name in task_names if task_name not in mapping]
    if unknown_tasks:
        raise ValueError(f"Tasks not configured in task_object_names.json: {unknown_tasks}")

    requested_objects = _parse_csv(args.objects)
    out_root = os.path.abspath(args.out_root or _default_out_root())
    run_records = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    env = VoxPoserRLBench(visualizer=None, camera_resolution=args.camera_resolution)
    try:
        env.enable_perception_backend(
            backend_type=args.backend_type,
            fallback_to_oracle=args.fallback_to_oracle,
            dump_root=out_root,
            cameras=tuple(_parse_csv(args.cameras) or ["front", "overhead", "wrist"]),
            use_openai_vlm=args.use_openai_vlm,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            vlm_api_mode=args.vlm_api_mode,
            vlm_api_key_env=args.vlm_api_key_env,
            vlm_reasoning_effort=args.vlm_reasoning_effort,
            vlm_max_phrases=args.vlm_max_phrases,
            vlm_phrase_count=args.vlm_phrase_count,
            timeout=args.timeout,
        )

        for task_name in task_names:
            task_cls = _task_class(task_name)
            objects = _object_list(task_name, mapping, requested_objects)
            env.load_task(task_cls)
            for episode_idx in range(args.episodes):
                descriptions, _ = env.reset()
                instruction = args.instruction or (descriptions[0] if descriptions else task_name)
                if hasattr(env.perception_backend, "set_vlm_instruction"):
                    env.perception_backend.set_vlm_instruction(instruction)

                for object_name in objects:
                    record = {
                        "task": task_name,
                        "episode": episode_idx,
                        "object": object_name,
                        "instruction": instruction,
                        "status": "started",
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    try:
                        points, _ = env.get_3d_obs_by_name(object_name)
                        record.update(
                            {
                                "status": "ok",
                                "point_count": int(len(points)),
                                "center": _center(points),
                            }
                        )
                        print(
                            "[task-benchmark] "
                            f"{task_name} episode={episode_idx} object={object_name!r} "
                            f"points={record['point_count']}"
                        )
                    except Exception as exc:
                        record.update({"status": "error", "error": str(exc)})
                        print(
                            "[task-benchmark] ERROR "
                            f"{task_name} episode={episode_idx} object={object_name!r}: {exc}"
                        )
                        if not args.keep_going:
                            raise
                    run_records.append(record)
    finally:
        env.shutdown()

    run_dir = os.path.join(out_root, "_benchmark")
    _save_json(
        os.path.join(run_dir, "run_records.json"),
        {
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "out_root": out_root,
            "tasks": task_names,
            "episodes": args.episodes,
            "records": run_records,
        },
    )

    bench_args = _benchmark_args(
        out_root=out_root,
        out_dir=os.path.join(run_dir, "perception_metrics"),
        args=args,
    )
    try:
        benchmark_result = run_benchmark(bench_args)
    except ValueError as exc:
        if "No summary.json files found" not in str(exc):
            raise
        benchmark_result = _empty_benchmark_result()
    _write_benchmark_outputs(benchmark_result, bench_args.out_dir)

    result = {
        "out_root": out_root,
        "run_records": os.path.join(run_dir, "run_records.json"),
        "benchmark_dir": bench_args.out_dir,
        "record_count": len(run_records),
        "ok_count": sum(record["status"] == "ok" for record in run_records),
        "error_count": sum(record["status"] == "error" for record in run_records),
        "benchmark": {
            "summary_count": benchmark_result["summary_count"],
            "pass_count": benchmark_result["pass_count"],
            "pass_rate": benchmark_result["pass_rate"],
            "oracle_count": benchmark_result["oracle_count"],
            "oracle_pass_rate": benchmark_result["oracle_pass_rate"],
            "mean_center_dist_m": benchmark_result["mean_center_dist_m"],
            "mean_mask_iou": benchmark_result["mean_mask_iou"],
        },
    }
    _save_json(os.path.join(run_dir, "result.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a live perception benchmark on RLBench tasks configured in "
            "envs/task_object_names.json. This resets each task and detects each exposed object."
        )
    )
    parser.add_argument("--tasks", help="Comma-separated task names. Defaults to all configured tasks.")
    parser.add_argument("--objects", help="Comma-separated exposed object names. Defaults to all objects per task.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--cameras", default="front,overhead,wrist")
    parser.add_argument("--out-root", help="Directory for generated perception dumps.")
    parser.add_argument("--backend-type", choices=("inprocess", "subprocess"), default="inprocess")
    parser.add_argument("--fallback-to-oracle", action="store_true")
    parser.add_argument("--use-openai-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="gpt-5.4")
    parser.add_argument("--vlm-base-url")
    parser.add_argument("--vlm-api-mode", default="responses")
    parser.add_argument("--vlm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--vlm-reasoning-effort", default="low")
    parser.add_argument("--vlm-max-phrases", type=int, default=2)
    parser.add_argument("--vlm-phrase-count", type=int, default=4)
    parser.add_argument("--instruction", help="Override VLM instruction for all detections.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--center-pass-threshold", type=float, default=0.08)
    parser.add_argument("--min-pred-points", type=int, default=20)
    parser.add_argument("--group-by", action="append", help="Aggregation fields for final metrics.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after per-object failures.")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.camera_resolution < 1:
        raise ValueError("--camera-resolution must be >= 1")
    if args.vlm_max_phrases < 1:
        raise ValueError("--vlm-max-phrases must be >= 1")
    if args.use_openai_vlm and not os.environ.get(args.vlm_api_key_env):
        raise ValueError(
            f"--use-openai-vlm requires environment variable {args.vlm_api_key_env}. "
            f"Set it with: export {args.vlm_api_key_env}=..."
        )
    if args.use_openai_vlm:
        _check_tcp_endpoint(args.vlm_base_url)

    result = run_live_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
