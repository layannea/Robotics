import argparse
import contextlib
import csv
import json
import os
import re
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import openai
import rlbench.tasks as tasks

from arguments import get_config
from envs.rlbench_env import VoxPoserRLBench
from interfaces import setup_LMP
from utils import set_lmp_objects


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "task",
        "episode",
        "status",
        "success",
        "task_success",
        "task_terminate",
        "latest_reward",
        "latest_terminate",
        "duration_s",
        "instruction",
        "error_type",
        "error",
        "attempt_count",
        "first_attempt_success",
        "self_repair_used",
        "self_repair_success",
        "log_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _snake_to_camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) == 1 and items[0].lower() == "all":
        return None
    return items


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
    return os.path.join(SRC_DIR, "planning_dumps", "oracle_planning", stamp)


def _configure_openai(args):
    api_key = os.environ.get(args.openai_api_key_env)
    if args.openai_api_key:
        api_key = args.openai_api_key
    if not api_key:
        raise ValueError(
            f"Environment variable {args.openai_api_key_env} is not set. "
            "Pass --openai-api-key or export the variable before running."
        )
    openai.api_key = api_key
    if args.openai_base_url:
        openai.api_base = args.openai_base_url.rstrip("/")


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
            f"Cannot connect to OpenAI-compatible base URL {base_url!r} "
            f"at {parsed.hostname}:{port}."
        ) from exc


def _load_config(args):
    config_path = os.path.abspath(args.config)
    config = get_config(config_path=config_path)
    config["lmp_config"]["env"]["visualize"] = bool(args.visualize)
    for lmp_cfg in config["lmp_config"]["lmps"].values():
        if args.model:
            lmp_cfg["model"] = args.model
        if args.max_tokens is not None:
            lmp_cfg["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            lmp_cfg["temperature"] = args.temperature
        if args.load_cache is not None:
            lmp_cfg["load_cache"] = args.load_cache
    return config


def _success_state(env: VoxPoserRLBench) -> Dict[str, Any]:
    task_success = None
    task_terminate = None
    try:
        task_success, task_terminate = env.task._task.success()
    except Exception:
        pass
    latest_reward = env.latest_reward
    latest_terminate = env.latest_terminate
    success = bool(task_success) or latest_reward == 1
    return {
        "success": success,
        "task_success": task_success,
        "task_terminate": task_terminate,
        "latest_reward": latest_reward,
        "latest_terminate": latest_terminate,
    }


def _configure_vision_backend(env: VoxPoserRLBench, args, out_root: str):
    if args.vision == "oracle":
        env.disable_perception_backend()
        return
    env.enable_perception_backend(
        backend_type=args.backend_type,
        fallback_to_oracle=args.fallback_to_oracle,
        dump_root=os.path.join(out_root, "perception_dumps"),
        cameras=tuple(_parse_csv(args.cameras) or ["front", "overhead", "wrist"]),
        use_openai_vlm=args.use_openai_vlm,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        vlm_api_mode=args.vlm_api_mode,
        vlm_api_key_env=args.vlm_api_key_env,
        vlm_reasoning_effort=args.vlm_reasoning_effort,
        vlm_max_phrases=args.vlm_max_phrases,
        vlm_phrase_count=args.vlm_phrase_count,
        vlm_use_candidate_scorer=not args.disable_vlm_candidate_scorer,
        timeout=args.perception_timeout,
    )


def _oracle_snapshot(env: VoxPoserRLBench, objects: List[str]) -> Dict[str, Any]:
    snapshot = {}
    for object_name in objects:
        try:
            points, _ = env.get_3d_obs_by_name(object_name)
            center = points.mean(axis=0).tolist() if len(points) else None
            snapshot[object_name] = {
                "status": "ok",
                "point_count": int(len(points)),
                "center": center,
            }
        except Exception as exc:
            snapshot[object_name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return snapshot


def _clear_lmp_histories(lmps: Dict[str, Any]):
    for lmp in lmps.values():
        if hasattr(lmp, "clear_exec_hist"):
            lmp.clear_exec_hist()


def _set_lmp_feedback(lmps: Dict[str, Any], feedback_context: str):
    for lmp in lmps.values():
        if hasattr(lmp, "set_feedback_context"):
            lmp.set_feedback_context(feedback_context)


def _clear_lmp_feedback(lmps: Dict[str, Any]):
    for lmp in lmps.values():
        if hasattr(lmp, "clear_feedback_context"):
            lmp.clear_feedback_context()


def _set_planning_verifier_enabled(lmps: Dict[str, Any], enabled: bool):
    for lmp in lmps.values():
        if hasattr(lmp, "set_planning_verifier_enabled"):
            lmp.set_planning_verifier_enabled(enabled)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _compact_line(line: str, max_len: int = 220) -> str:
    line = _strip_ansi(line).strip()
    line = re.sub(r"\s+", " ", line)
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def _tail_matching_lines(log_path: str, patterns: List[str], limit: int = 4) -> List[str]:
    if not os.path.exists(log_path):
        return []
    regexes = [re.compile(pattern) for pattern in patterns]
    matches = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = _strip_ansi(raw_line)
            if any(regex.search(line) for regex in regexes):
                matches.append(_compact_line(line))
    return matches[-limit:]


def _format_feedback_context(
    *,
    env: VoxPoserRLBench,
    instruction: str,
    objects: List[str],
    previous_attempt: Dict[str, Any],
) -> str:
    """Build short, code-comment feedback for one execution-feedback retry."""
    log_path = previous_attempt.get("log_path", "")
    controller_lines = _tail_matching_lines(
        log_path,
        [
            r"Ignoring failed arm action",
            r"target outside",
            r"path could not be found",
            r"failed waypoint",
        ],
    )
    waypoint_lines = _tail_matching_lines(
        log_path,
        [r"last waypoint", r"target:", r"reached last waypoint", r"dist2target"],
        limit=5,
    )
    indexing_lines = _tail_matching_lines(
        log_path,
        [r"IndexingWrapper.*Warning", r"index was changed"],
        limit=3,
    )

    ee_pos = None
    try:
        ee_pos = env.get_ee_pos().round(4).tolist()
    except Exception:
        pass

    lines = [
        "# Execution feedback from the previous attempt:",
        f"# - Original instruction: {instruction}",
        f"# - Available objects: {objects}",
        (
            "# - Previous outcome: "
            f"status={previous_attempt.get('status')}, "
            f"success={previous_attempt.get('success')}, "
            f"reward={previous_attempt.get('latest_reward')}, "
            f"terminate={previous_attempt.get('latest_terminate')}"
        ),
        f"# - Workspace min: {env.workspace_bounds_min.round(4).tolist()}",
        f"# - Workspace max: {env.workspace_bounds_max.round(4).tolist()}",
    ]
    if ee_pos is not None:
        lines.append(f"# - Current end-effector position: {ee_pos}")
    if previous_attempt.get("error_type"):
        lines.append(
            "# - Runtime error: "
            f"{previous_attempt.get('error_type')}: {_compact_line(previous_attempt.get('error', ''))}"
        )
    if controller_lines:
        lines.append("# - Controller/runtime signals:")
        lines.extend(f"#   * {line}" for line in controller_lines)
    if waypoint_lines:
        lines.append("# - Recent waypoint/target signals:")
        lines.extend(f"#   * {line}" for line in waypoint_lines)
    if indexing_lines:
        lines.append("# - Voxel indexing signals:")
        lines.extend(f"#   * {line}" for line in indexing_lines)
    lines.extend(
        [
            "# Revise the plan/code for the same instruction using this feedback.",
            "# Avoid repeating unreachable targets or object choices that caused the previous failure.",
            "# Return executable Python code only, following the same VoxPoser APIs.",
        ]
    )
    return "\n".join(lines)


def _run_planning_attempt(
    *,
    env: VoxPoserRLBench,
    lmps: Dict[str, Any],
    voxposer_ui,
    objects: List[str],
    instruction: str,
    log_path: str,
    attempt_index: int,
) -> Dict[str, Any]:
    attempt = {
        "attempt": attempt_index,
        "log_path": log_path,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    start_time = time.time()
    _clear_lmp_histories(lmps)
    set_lmp_objects(lmps, objects)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_f:
            with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                voxposer_ui(instruction)
        attempt.update(_success_state(env))
        attempt["status"] = "ok"
    except Exception as exc:
        attempt.update(_success_state(env))
        attempt.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n\n# BENCHMARK ERROR\n")
            log_f.write(attempt["traceback"])
    finally:
        attempt["duration_s"] = time.time() - start_time
        attempt["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return attempt


def _summarize(records: List[Dict[str, Any]], args) -> Dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    error_records = [record for record in records if record.get("status") == "error"]
    success_records = [record for record in ok_records if record.get("success")]
    first_attempt_success_records = [
        record for record in records if record.get("first_attempt_success")
    ]
    self_repair_records = [record for record in records if record.get("self_repair_used")]
    self_repair_success_records = [
        record for record in records if record.get("self_repair_success")
    ]
    by_task = []
    for task in sorted({record["task"] for record in records}):
        task_records = [record for record in records if record["task"] == task]
        task_ok = [record for record in task_records if record.get("status") == "ok"]
        task_success = [record for record in task_ok if record.get("success")]
        task_first_attempt_success = [
            record for record in task_records if record.get("first_attempt_success")
        ]
        task_self_repair = [record for record in task_records if record.get("self_repair_used")]
        task_self_repair_success = [
            record for record in task_records if record.get("self_repair_success")
        ]
        by_task.append(
            {
                "task": task,
                "episode_count": len(task_records),
                "ok_count": len(task_ok),
                "error_count": len(task_records) - len(task_ok),
                "success_count": len(task_success),
                "success_rate": (len(task_success) / len(task_records)) if task_records else None,
                "first_attempt_success_count": len(task_first_attempt_success),
                "first_attempt_success_rate": (
                    len(task_first_attempt_success) / len(task_records)
                    if task_records
                    else None
                ),
                "self_repair_used_count": len(task_self_repair),
                "self_repair_success_count": len(task_self_repair_success),
                "self_repair_success_rate": (
                    len(task_self_repair_success) / len(task_self_repair)
                    if task_self_repair
                    else None
                ),
            }
        )
    return {
        "vision": "rlbench_oracle_masks" if args.vision == "oracle" else "vlm_grounded_sam2",
        "debug_mode": args.debug,
        "planning_verifier_enabled": not args.disable_planning_verifier,
        "record_count": len(records),
        "ok_count": len(ok_records),
        "error_count": len(error_records),
        "success_count": len(success_records),
        "success_rate": (len(success_records) / len(records)) if records else None,
        "first_attempt_success_count": len(first_attempt_success_records),
        "first_attempt_success_rate": (
            len(first_attempt_success_records) / len(records) if records else None
        ),
        "self_repair_retries": args.self_repair_retries,
        "self_repair_used_count": len(self_repair_records),
        "self_repair_success_count": len(self_repair_success_records),
        "self_repair_success_rate": (
            len(self_repair_success_records) / len(self_repair_records)
            if self_repair_records
            else None
        ),
        "by_task": by_task,
    }


def run_benchmark(args) -> Dict[str, Any]:
    _check_tcp_endpoint(args.openai_base_url)
    _configure_openai(args)

    mapping = _load_task_object_mapping()
    task_names = _parse_csv(args.tasks) or list(mapping.keys())
    unknown_tasks = [task_name for task_name in task_names if task_name not in mapping]
    if unknown_tasks:
        raise ValueError(f"Tasks not configured in task_object_names.json: {unknown_tasks}")

    out_root = os.path.abspath(args.out_root or _default_out_root())
    logs_dir = os.path.join(out_root, "logs")
    records = []
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    config = _load_config(args)
    env = VoxPoserRLBench(visualizer=None, camera_resolution=args.camera_resolution)
    _configure_vision_backend(env, args, out_root)

    try:
        lmps, _ = setup_LMP(env, config, debug=args.debug)
        _set_planning_verifier_enabled(lmps, not args.disable_planning_verifier)
        voxposer_ui = lmps["plan_ui"]

        for task_name in task_names:
            task_cls = _task_class(task_name)
            env.load_task(task_cls)
            objects = env.get_object_names()
            set_lmp_objects(lmps, objects)

            for episode_idx in range(args.episodes):
                log_path = os.path.join(logs_dir, task_name, f"episode_{episode_idx:03d}.log")
                record = {
                    "task": task_name,
                    "episode": episode_idx,
                    "objects": objects,
                    "status": "started",
                    "log_path": log_path,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                start_time = time.time()
                try:
                    descriptions, _ = env.reset()
                    instruction = args.instruction or (descriptions[0] if descriptions else task_name)
                    if env.perception_backend is not None and hasattr(env.perception_backend, "set_vlm_instruction"):
                        env.perception_backend.set_vlm_instruction(instruction)
                    record["instruction"] = instruction
                    if not args.skip_oracle_precheck:
                        record["oracle_objects"] = _oracle_snapshot(env, objects)
                    _clear_lmp_feedback(lmps)
                    attempts = []

                    max_attempts = max(1, int(args.self_repair_retries) + 1)
                    for attempt_idx in range(max_attempts):
                        if attempt_idx == 0:
                            attempt_log_path = log_path
                        else:
                            attempt_log_path = os.path.join(
                                logs_dir,
                                task_name,
                                f"episode_{episode_idx:03d}_repair_{attempt_idx:02d}.log",
                            )

                        attempt = _run_planning_attempt(
                            env=env,
                            lmps=lmps,
                            voxposer_ui=voxposer_ui,
                            objects=objects,
                            instruction=instruction,
                            log_path=attempt_log_path,
                            attempt_index=attempt_idx,
                        )
                        attempts.append(attempt)
                        if attempt.get("success"):
                            break
                        if attempt_idx >= max_attempts - 1:
                            break

                        feedback_context = _format_feedback_context(
                            env=env,
                            instruction=instruction,
                            objects=objects,
                            previous_attempt=attempt,
                        )
                        attempt["self_repair_feedback"] = feedback_context
                        _set_lmp_feedback(lmps, feedback_context)

                    final_attempt = attempts[-1]
                    record["attempts"] = attempts
                    record["attempt_count"] = len(attempts)
                    record["first_attempt_success"] = bool(attempts[0].get("success"))
                    record["self_repair_used"] = len(attempts) > 1
                    record["self_repair_success"] = (
                        bool(final_attempt.get("success")) and not bool(attempts[0].get("success"))
                    )
                    record["log_path"] = final_attempt.get("log_path", log_path)
                    record.update({k: v for k, v in final_attempt.items() if k != "attempts"})
                    record["duration_s"] = time.time() - start_time
                    print(
                        "[planning-benchmark] "
                        f"{task_name} episode={episode_idx} "
                        f"success={record['success']} reward={record['latest_reward']} "
                        f"terminate={record['latest_terminate']} "
                        f"attempts={record['attempt_count']}"
                    )
                    if record["status"] == "error" and not args.keep_going:
                        raise RuntimeError(
                            f"{task_name} episode={episode_idx} failed after "
                            f"{record['attempt_count']} attempt(s): "
                            f"{record.get('error_type')}: {record.get('error')}"
                        )
                except Exception as exc:
                    duration_s = time.time() - start_time
                    record.update(_success_state(env))
                    record.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "duration_s": duration_s,
                        }
                    )
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as log_f:
                        log_f.write("\n\n# BENCHMARK ERROR\n")
                        log_f.write(record["traceback"])
                    print(
                        "[planning-benchmark] ERROR "
                        f"{task_name} episode={episode_idx}: {type(exc).__name__}: {exc}"
                    )
                    if not args.keep_going:
                        raise
                finally:
                    record.setdefault("duration_s", time.time() - start_time)
                    record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    _clear_lmp_feedback(lmps)
                    records.append(record)
    finally:
        env.shutdown()

    summary = _summarize(records, args)
    run_records_path = os.path.join(out_root, "run_records.json")
    summary_path = os.path.join(out_root, "planning_benchmark.json")
    details_path = os.path.join(out_root, "details.csv")
    payload = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_root": out_root,
        "config": {
            "tasks": task_names,
            "episodes": args.episodes,
            "model": args.model,
            "openai_base_url": args.openai_base_url,
            "camera_resolution": args.camera_resolution,
            "self_repair_retries": args.self_repair_retries,
            "planning_verifier_enabled": not args.disable_planning_verifier,
            "vision": args.vision,
            "backend_type": args.backend_type if args.vision == "perception" else None,
            "use_openai_vlm": bool(args.use_openai_vlm) if args.vision == "perception" else None,
            "vlm_model": args.vlm_model if args.vision == "perception" else None,
            "vlm_candidate_scorer_enabled": (
                not args.disable_vlm_candidate_scorer if args.vision == "perception" else None
            ),
        },
        "summary": summary,
        "records": records,
    }
    _save_json(run_records_path, payload)
    _save_json(summary_path, summary)
    _write_csv(details_path, records)

    return {
        "out_root": out_root,
        "run_records": run_records_path,
        "benchmark": summary_path,
        "details_csv": details_path,
        **summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark VoxPoser planning with RLBench oracle object masks. "
            "This isolates LMP planning/execution from the learned perception backend."
        )
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task names, or all. Default: all configured tasks.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--instruction", default=None, help="Override RLBench description for every run.")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--config", default=os.path.join(SRC_DIR, "configs", "rlbench_config.yaml"))
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument(
        "--vision",
        choices=("oracle", "perception"),
        default="oracle",
        help="Use RLBench oracle masks or the learned/VLM perception backend.",
    )
    parser.add_argument("--backend-type", choices=("inprocess", "subprocess"), default="inprocess")
    parser.add_argument("--fallback-to-oracle", action="store_true")
    parser.add_argument("--cameras", default="front,overhead,wrist")
    parser.add_argument("--use-openai-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="gpt-5.4")
    parser.add_argument("--vlm-base-url")
    parser.add_argument("--vlm-api-mode", default="responses")
    parser.add_argument("--vlm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--vlm-reasoning-effort", default="low")
    parser.add_argument("--vlm-max-phrases", type=int, default=2)
    parser.add_argument("--vlm-phrase-count", type=int, default=4)
    parser.add_argument("--disable-vlm-candidate-scorer", action="store_true")
    parser.add_argument("--perception-timeout", type=int, default=180)
    parser.add_argument("--model", default=None, help="Override every LMP model in the config.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--load-cache", dest="load_cache", action="store_true", default=None)
    parser.add_argument("--no-load-cache", dest="load_cache", action="store_false")
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--debug", action="store_true", help="Generate code but comment out execute(...) calls.")
    parser.add_argument("--visualize", action="store_true", help="Enable VoxPoser voxel visualizer if one is attached.")
    parser.add_argument("--skip-oracle-precheck", action="store_true")
    parser.add_argument(
        "--self-repair-retries",
        type=int,
        default=0,
        help=(
            "Number of within-episode execution-feedback retries after a failed "
            "planning attempt. Default 0 preserves the one-shot benchmark."
        ),
    )
    parser.add_argument(
        "--disable-planning-verifier",
        action="store_true",
        help="Disable lightweight generated-code verifier/repair before execution.",
    )
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
