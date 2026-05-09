import argparse
import glob
import itertools
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from .benchmark_perception import _save_json, _write_aggregate_csv, run_benchmark
    from .distractor_verifier import annotate_candidate_with_distractors, distractor_masks_by_camera
    from .evaluate_perception import _write_csv
    from .role_aware import query_name_from_phrase_candidates, role_for_query, vlm_summary_for_phrase_candidates
    from .scorer_v2 import SCORER_VERSION, score_camera_candidate_v2, score_phrase_summary_v2
except ImportError:
    from benchmark_perception import _save_json, _write_aggregate_csv, run_benchmark
    from distractor_verifier import annotate_candidate_with_distractors, distractor_masks_by_camera
    from evaluate_perception import _write_csv
    from role_aware import query_name_from_phrase_candidates, role_for_query, vlm_summary_for_phrase_candidates
    from scorer_v2 import SCORER_VERSION, score_camera_candidate_v2, score_phrase_summary_v2


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _safe_rel(path: str, roots: List[str]) -> str:
    abs_path = os.path.abspath(path)
    for root in sorted([os.path.abspath(root) for root in roots], key=len, reverse=True):
        try:
            rel = os.path.relpath(abs_path, root)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return rel
    return os.path.basename(abs_path)


def _image_shape(rgb_path: Optional[str]) -> Optional[List[int]]:
    if not rgb_path or not os.path.exists(rgb_path):
        return None
    try:
        from PIL import Image

        with Image.open(rgb_path) as image:
            width, height = image.size
        return [height, width]
    except Exception:
        return None


def _infer_candidate_path_from_overlay(overlay_path: Optional[str], suffix: str) -> Optional[str]:
    if not overlay_path:
        return None
    if overlay_path.endswith("_overlay.png"):
        path = overlay_path[: -len("_overlay.png")] + suffix
        return path if os.path.exists(path) else None
    return None


def _passes_hard_filters(candidate: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    point_count = int(candidate.get("object_point_count") or 0)
    min_points = int(config.get("min_points") or 20)
    if point_count < min_points:
        return False, "too_few_points"
    max_points = config.get("max_points")
    if max_points is not None and point_count > int(max_points):
        return False, "too_many_points"
    box_score = candidate.get("box_score")
    min_score = config.get("min_score")
    if min_score is not None and box_score is not None and float(box_score) < float(min_score):
        return False, "low_score"
    center = candidate.get("object_point_center")
    max_center_z = config.get("max_center_z")
    if max_center_z is not None and center is not None and float(center[2]) > float(max_center_z):
        return False, "center_z_too_high"
    points_path = candidate.get("object_points")
    if not points_path or not os.path.exists(points_path):
        return False, "missing_points_file"
    return True, None


def _original_camera_option(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not result.get("object_points") or not os.path.exists(result["object_points"]):
        return None
    option = dict(result)
    option["candidate_kind"] = "original"
    option["source_box_index"] = result.get("selected_box_index")
    option["box_area_frac"] = _box_area_frac(result.get("box"), result.get("rgb"))
    option["image_shape"] = result.get("image_shape") or _image_shape(result.get("rgb"))
    return option


def _fallback_options(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = []
    fallback = result.get("candidate_fallback") or {}
    for candidate in fallback.get("candidates") or []:
        overlay = candidate.get("overlay")
        points_path = _infer_candidate_path_from_overlay(overlay, "_points.npy")
        mask_path = _infer_candidate_path_from_overlay(overlay, "_mask.npy")
        if not points_path:
            continue
        option = dict(candidate)
        option.update(
            {
                "candidate_kind": "fallback",
                "camera": result.get("camera"),
                "rgb": result.get("rgb"),
                "point_cloud": result.get("point_cloud"),
                "object_points": points_path,
                "mask": mask_path,
                "image_shape": result.get("image_shape") or _image_shape(result.get("rgb")),
            }
        )
        options.append(option)
    return options


def _bank_options(result: Dict[str, Any], summary_path: str, bank_name: str) -> List[Dict[str, Any]]:
    bank_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)), bank_name, "candidate_bank.json")
    if not os.path.exists(bank_path):
        return []
    try:
        bank = _load_json(bank_path)
    except Exception:
        return []
    options = []
    for candidate in bank.get("candidates") or []:
        if candidate.get("camera") != result.get("camera"):
            continue
        if not candidate.get("object_points") or not os.path.exists(candidate["object_points"]):
            continue
        option = dict(candidate)
        option["candidate_kind"] = "bank"
        option["candidate_bank"] = bank_path
        options.append(option)
    return options


def _box_area_frac(box, rgb_path: Optional[str]) -> Optional[float]:
    if box is None or not rgb_path:
        return None
    shape = _image_shape(rgb_path)
    if not shape:
        return None
    height, width = float(shape[0]), float(shape[1])
    area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
    return area / max(1.0, height * width)


def _candidate_to_camera_result(option: Dict[str, Any], score_info: Dict[str, Any], used: bool, reject_reason: Optional[str]) -> Dict[str, Any]:
    return {
        "camera": option.get("camera"),
        "accepted": bool(used),
        "used_for_fusion": bool(used),
        "reject_reason": reject_reason,
        "replay_candidate_kind": option.get("candidate_kind"),
        "replay_score_info": score_info,
        "rgb": option.get("rgb"),
        "point_cloud": option.get("point_cloud"),
        "selected_box_index": option.get("source_box_index", option.get("selected_box_index")),
        "box": option.get("box"),
        "box_score": option.get("box_score", option.get("score")),
        "label": option.get("label"),
        "mask_pixels": option.get("mask_pixels"),
        "object_point_count": option.get("object_point_count"),
        "object_point_center": option.get("object_point_center"),
        "mask": option.get("mask"),
        "object_points": option.get("object_points"),
        "overlay": option.get("overlay"),
        "image_shape": option.get("image_shape"),
        "max_distractor_iou": option.get("max_distractor_iou"),
        "max_distractor_object": option.get("max_distractor_object"),
        "max_distractor_relation": option.get("max_distractor_relation"),
        "max_negative_distractor_iou": option.get("max_negative_distractor_iou"),
        "max_negative_distractor_object": option.get("max_negative_distractor_object"),
        "max_negative_distractor_center_dist_m": option.get("max_negative_distractor_center_dist_m"),
        "distractor_overlaps": option.get("distractor_overlaps"),
        "negative_distractor_overlaps": option.get("negative_distractor_overlaps"),
    }


def _replay_camera_results(
    summary_path: str,
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    camera_score_threshold: float,
    max_options_per_camera: int,
    query_name: str,
    phrase: str,
    phrase_rank: int,
    bank_name: str,
    use_candidate_bank: bool,
    use_distractor_verifier: bool,
    distractors_by_camera: Optional[Dict[str, List[Dict[str, str]]]] = None,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    distractors_by_camera = distractors_by_camera or {}
    groups = []
    for result in summary.get("camera_results") or []:
        options = _bank_options(result, summary_path, bank_name) if use_candidate_bank else []
        original = _original_camera_option(result)
        if original:
            options.append(original)
        options.extend(_fallback_options(result))

        scored = []
        for option in options:
            if use_distractor_verifier:
                option = annotate_candidate_with_distractors(
                    dict(option),
                    distractors_by_camera,
                    query_object=query_name,
                    query_role=config.get("role_category"),
                )
            score_info = score_camera_candidate_v2(option, config, manifest)
            hard_pass, hard_reason = _passes_hard_filters(option, config)
            if hard_pass and score_info["score"] >= camera_score_threshold:
                scored.append((score_info["score"], option, score_info, hard_reason))

        if not scored:
            groups.append({"source": result, "options": []})
        else:
            scored.sort(key=lambda item: item[0], reverse=True)
            groups.append({"source": result, "options": scored[:max_options_per_camera]})

    best = None
    choice_ranges = [range(-1, len(group["options"])) for group in groups]
    for choice in itertools.product(*choice_ranges):
        if not choice or all(index < 0 for index in choice):
            continue
        replayed_results = []
        fused_chunks = []
        for group, option_index in zip(groups, choice):
            if option_index < 0:
                replayed = dict(group["source"])
                replayed["accepted"] = False
                replayed["used_for_fusion"] = False
                replayed["reject_reason"] = "replay_not_selected"
                replayed["replay_score_info"] = {
                    "version": SCORER_VERSION,
                    "score": -100.0,
                    "reasons": ["replay_not_selected"],
                }
                replayed_results.append(replayed)
                continue
            _, option, score_info, _ = group["options"][option_index]
            replayed_results.append(_candidate_to_camera_result(option, score_info, True, None))
            fused_chunks.append(_finite_points(np.load(option["object_points"])))

        fused_points = (
            _finite_points(np.concatenate(fused_chunks, axis=0))
            if fused_chunks
            else np.empty((0, 3), dtype=np.float32)
        )
        replay_summary = dict(summary)
        replay_summary.update(
            {
                "camera_results": replayed_results,
                "fused_point_count": int(len(fused_points)),
                "fused_object_point_center": _center(fused_points),
            }
        )
        phrase_score = score_phrase_summary_v2(
            query_name=query_name,
            phrase=phrase,
            phrase_rank=phrase_rank,
            config=config,
            summary=replay_summary,
            points=fused_points,
        )
        key = (phrase_score["score"], len(fused_chunks))
        if best is None or key > best[0]:
            best = (key, replayed_results, [chunk for chunk in fused_chunks], phrase_score)

    if best is not None:
        _, replayed_results, fused_points, _ = best
        return replayed_results, fused_points

    replayed_results = []
    for group in groups:
        replayed = dict(group["source"])
        replayed["accepted"] = False
        replayed["used_for_fusion"] = False
        replayed["reject_reason"] = group["source"].get("reject_reason") or group["source"].get("reason") or "no_replay_candidates"
        replayed["replay_score_info"] = {"version": SCORER_VERSION, "score": -100.0, "reasons": ["no_replay_candidates"]}
        replayed_results.append(replayed)
    return replayed_results, []


def _summary_config(
    summary: Dict[str, Any],
    role_info: Optional[Dict[str, Any]] = None,
    role_verifier_mode: str = "soft",
) -> Dict[str, Any]:
    config = {
        "selection": summary.get("selection"),
        "min_points": summary.get("min_points", 20),
        "max_points": summary.get("max_points"),
        "min_score": summary.get("min_score"),
        "max_center_z": summary.get("max_center_z"),
    }
    if role_info:
        config.update(role_info)
    config["role_verifier_mode"] = role_verifier_mode
    return config


def _replay_phrase_summary(
    summary_path: str,
    query_name: str,
    phrase: str,
    phrase_rank: int,
    out_dir: str,
    camera_score_threshold: float,
    max_options_per_camera: int,
    bank_name: str,
    use_candidate_bank: bool,
    use_distractor_verifier: bool,
    role_info: Optional[Dict[str, Any]],
    role_verifier_mode: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    summary = _load_json(summary_path)
    manifest_path = summary.get("manifest")
    manifest = _load_json(manifest_path) if manifest_path and os.path.exists(manifest_path) else {}
    config = _summary_config(summary, role_info, role_verifier_mode)
    distractors = distractor_masks_by_camera(summary_path) if use_distractor_verifier else {}
    replayed_results, fused_chunks = _replay_camera_results(
        summary_path=summary_path,
        summary=summary,
        manifest=manifest,
        config=config,
        camera_score_threshold=camera_score_threshold,
        max_options_per_camera=max_options_per_camera,
        query_name=query_name,
        phrase=phrase,
        phrase_rank=phrase_rank,
        bank_name=bank_name,
        use_candidate_bank=use_candidate_bank,
        use_distractor_verifier=use_distractor_verifier,
        distractors_by_camera=distractors,
    )
    fused_points = _finite_points(np.concatenate(fused_chunks, axis=0)) if fused_chunks else np.empty((0, 3), dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    fused_points_path = os.path.join(out_dir, "fused_object_points.npy")
    np.save(fused_points_path, fused_points)
    replay_summary = dict(summary)
    replay_summary.update(
        {
            "replay": True,
            "replay_scorer_version": SCORER_VERSION,
            "replay_source_summary": summary_path,
            "replay_camera_score_threshold": camera_score_threshold,
            "replay_bank_name": bank_name,
            "replay_use_candidate_bank": use_candidate_bank,
            "replay_use_distractor_verifier": use_distractor_verifier,
            "replay_distractor_count": sum(len(items) for items in distractors.values()),
            "replay_role": config.get("role"),
            "replay_role_category": config.get("role_category"),
            "camera_results": replayed_results,
            "fused_object_points": fused_points_path,
            "fused_point_count": int(len(fused_points)),
            "fused_object_point_center": _center(fused_points),
        }
    )
    replay_summary_path = os.path.join(out_dir, "summary.json")
    with open(replay_summary_path, "w", encoding="utf-8") as f:
        json.dump(replay_summary, f, indent=2)

    phrase_score = score_phrase_summary_v2(
        query_name=query_name,
        phrase=phrase,
        phrase_rank=phrase_rank,
        config=config,
        summary=replay_summary,
        points=fused_points,
    )
    return replay_summary_path, replay_summary, phrase_score


def _replay_phrase_group(
    candidates_path: str,
    roots: List[str],
    out_root: str,
    camera_score_threshold: float,
    max_options_per_camera: int,
    bank_name: str,
    use_candidate_bank: bool,
    use_role_aware: bool,
    use_distractor_verifier: bool,
    role_verifier_mode: str,
) -> Optional[Dict[str, Any]]:
    data = _load_json(candidates_path)
    query_name = query_name_from_phrase_candidates(candidates_path)
    role_info = None
    if use_role_aware:
        vlm_path = vlm_summary_for_phrase_candidates(candidates_path)
        vlm_summary = _load_json(vlm_path) if vlm_path and os.path.exists(vlm_path) else None
        role_info = role_for_query(query_name, vlm_summary)
    rel_parent = os.path.dirname(_safe_rel(candidates_path, roots))
    group_out_dir = os.path.join(out_root, "replayed", rel_parent)

    records = []
    for record in data.get("records") or []:
        summary_path = record.get("summary")
        if not summary_path or not os.path.exists(summary_path):
            replay_record = dict(record)
            replay_record["replay_status"] = "missing_summary"
            records.append(replay_record)
            continue
        phrase = record.get("phrase") or ""
        rank = int(record.get("rank") or 0)
        phrase_out_dir = os.path.join(group_out_dir, f"phrase_{rank:02d}_{_safe_name(phrase)}")
        try:
            replay_summary_path, replay_summary, score_info = _replay_phrase_summary(
                summary_path=summary_path,
                query_name=query_name,
                phrase=phrase,
                phrase_rank=rank,
                out_dir=phrase_out_dir,
                camera_score_threshold=camera_score_threshold,
                max_options_per_camera=max_options_per_camera,
                bank_name=bank_name,
                use_candidate_bank=use_candidate_bank,
                use_distractor_verifier=use_distractor_verifier,
                role_info=role_info,
                role_verifier_mode=role_verifier_mode,
            )
            replay_record = dict(record)
            replay_record.update(
                {
                    "replay_status": "scored",
                    "replay_summary": replay_summary_path,
                    "replay_score_info": score_info,
                    "replay_fused_point_count": replay_summary.get("fused_point_count"),
                    "replay_fused_object_point_center": replay_summary.get("fused_object_point_center"),
                }
            )
        except Exception as exc:
            replay_record = dict(record)
            replay_record.update({"replay_status": "error", "replay_error": str(exc)})
        records.append(replay_record)

    scored = [record for record in records if record.get("replay_status") == "scored"]
    if not scored:
        return None
    scored.sort(key=lambda record: record["replay_score_info"]["score"], reverse=True)
    selected = scored[0]
    for record in records:
        record["replay_selected"] = record is selected
        record.pop("selected", None)

    replay_candidates = {
        "type": "phrases_replay",
        "source_phrase_candidates": candidates_path,
        "scorer_version": SCORER_VERSION,
        "camera_score_threshold": camera_score_threshold,
        "role_info": role_info,
        "role_verifier_mode": role_verifier_mode,
        "use_distractor_verifier": use_distractor_verifier,
        "selected_phrase": selected.get("phrase"),
        "selected_replay_summary": selected.get("replay_summary"),
        "records": records,
    }
    os.makedirs(group_out_dir, exist_ok=True)
    replay_candidates_path = os.path.join(group_out_dir, "phrase_candidates_replay.json")
    with open(replay_candidates_path, "w", encoding="utf-8") as f:
        json.dump(replay_candidates, f, indent=2)

    return {
        "source": candidates_path,
        "query_name": query_name,
        "selected_summary": selected.get("replay_summary"),
        "selected_phrase": selected.get("phrase"),
        "selected_score": selected.get("replay_score_info", {}).get("score"),
        "role": role_info.get("role") if role_info else None,
        "role_category": role_info.get("role_category") if role_info else None,
        "replay_candidates": replay_candidates_path,
        "record_count": len(records),
    }


def _safe_name(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:80] or "candidate"


def _discover_phrase_candidates(roots: Iterable[str]) -> List[str]:
    paths = []
    for root in roots:
        paths.extend(glob.glob(os.path.join(root, "**", "phrase_candidates.json"), recursive=True))
    return sorted(os.path.abspath(path) for path in paths)


def _is_under_phrase_dir(summary_path: str) -> bool:
    return any(part.startswith("phrase_") for part in summary_path.replace(os.sep, "/").split("/"))


def _discover_passthrough_results(roots: Iterable[str]) -> List[str]:
    paths = []
    for root in roots:
        for summary_path in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
            if _is_under_phrase_dir(summary_path):
                continue
            paths.append(os.path.abspath(summary_path))
        paths.extend(os.path.abspath(path) for path in glob.glob(os.path.join(root, "**", "instances.json"), recursive=True))
    return sorted(set(paths))


def _write_benchmark(out_dir: str, result: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    _save_json(os.path.join(out_dir, "benchmark.json"), result)
    _write_csv(os.path.join(out_dir, "details.csv"), result["rows"])
    _write_aggregate_csv(os.path.join(out_dir, "aggregate.csv"), result["aggregate_rows"])


def replay_scorer_benchmark(args) -> Dict[str, Any]:
    roots = [os.path.abspath(root) for root in args.root]
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    replay_records = []
    selected_paths = []
    for candidates_path in _discover_phrase_candidates(roots):
        replay = _replay_phrase_group(
            candidates_path=candidates_path,
            roots=roots,
            out_root=out_root,
            camera_score_threshold=args.camera_score_threshold,
            max_options_per_camera=args.max_options_per_camera,
            bank_name=args.bank_name,
            use_candidate_bank=args.use_candidate_bank,
            use_role_aware=args.use_role_aware,
            use_distractor_verifier=args.use_distractor_verifier,
            role_verifier_mode=args.role_verifier_mode,
        )
        if replay and replay.get("selected_summary"):
            replay_records.append(replay)
            selected_paths.append(replay["selected_summary"])

    passthrough_paths = _discover_passthrough_results(roots)
    if args.include_passthrough:
        selected_paths.extend(passthrough_paths)

    selected_paths = sorted(dict.fromkeys(os.path.abspath(path) for path in selected_paths if path and os.path.exists(path)))
    replay_summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scorer_version": SCORER_VERSION,
        "roots": roots,
        "out_root": out_root,
        "camera_score_threshold": args.camera_score_threshold,
        "max_options_per_camera": args.max_options_per_camera,
        "bank_name": args.bank_name,
        "use_candidate_bank": args.use_candidate_bank,
        "use_role_aware": args.use_role_aware,
        "use_distractor_verifier": args.use_distractor_verifier,
        "role_verifier_mode": args.role_verifier_mode,
        "phrase_group_count": len(replay_records),
        "passthrough_count": len(passthrough_paths) if args.include_passthrough else 0,
        "selected_result_count": len(selected_paths),
        "selected_paths": selected_paths,
        "replay_records": replay_records,
    }
    _save_json(os.path.join(out_root, "replay_records.json"), replay_summary)

    bench_args = SimpleNamespace(
        root=[],
        summaries_glob=[],
        summary=selected_paths,
        selected_only=False,
        max_summaries=None,
        oracle_map=args.oracle_map,
        oracle_id=args.oracle_id,
        object_name=args.object_name,
        center_pass_threshold=args.center_pass_threshold,
        min_pred_points=args.min_pred_points,
        require_oracle=args.require_oracle,
        group_by=args.group_by,
        keep_going=args.keep_going,
    )
    benchmark = run_benchmark(bench_args)
    benchmark["replay"] = {
        "scorer_version": SCORER_VERSION,
        "camera_score_threshold": args.camera_score_threshold,
        "bank_name": args.bank_name,
        "use_candidate_bank": args.use_candidate_bank,
        "use_role_aware": args.use_role_aware,
        "use_distractor_verifier": args.use_distractor_verifier,
        "role_verifier_mode": args.role_verifier_mode,
        "replay_records": os.path.join(out_root, "replay_records.json"),
    }
    benchmark_dir = os.path.join(out_root, "benchmark")
    _write_benchmark(benchmark_dir, benchmark)

    result = {
        "out_root": out_root,
        "replay_records": os.path.join(out_root, "replay_records.json"),
        "benchmark_dir": benchmark_dir,
        "selected_result_count": len(selected_paths),
        "phrase_group_count": len(replay_records),
        "benchmark": {
            key: benchmark.get(key)
            for key in (
                "summary_count",
                "pass_count",
                "pass_rate",
                "oracle_count",
                "oracle_pass_rate",
                "mean_center_dist_m",
                "mean_mask_iou",
                "error_count",
            )
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay saved perception dumps with scorer v2 and benchmark the selected results.")
    parser.add_argument("--root", action="append", required=True, help="Saved perception dump root. Can be repeated.")
    parser.add_argument("--out-dir", required=True, help="Output directory for replay summaries and benchmark results.")
    parser.add_argument("--camera-score-threshold", type=float, default=0.2)
    parser.add_argument("--max-options-per-camera", type=int, default=4)
    parser.add_argument("--bank-name", default="candidate_bank")
    parser.add_argument("--use-candidate-bank", action="store_true", help="Include generated candidate_bank candidates in replay search.")
    parser.add_argument("--use-role-aware", action="store_true", help="Condition scorer v2 on VLM-derived object roles.")
    parser.add_argument("--use-distractor-verifier", action="store_true", help="Penalize candidates that overlap selected masks for sibling objects in the same observation.")
    parser.add_argument("--role-verifier-mode", choices=("soft", "strict"), default="soft")
    parser.add_argument("--no-passthrough", dest="include_passthrough", action="store_false", help="Do not include non-phrase summary.json or instances.json results.")
    parser.set_defaults(include_passthrough=True)
    parser.add_argument("--oracle-map")
    parser.add_argument("--oracle-id", action="append")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--center-pass-threshold", type=float, default=0.08)
    parser.add_argument("--min-pred-points", type=int, default=20)
    parser.add_argument("--require-oracle", action="store_true")
    parser.add_argument("--group-by", action="append")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    result = replay_scorer_benchmark(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
