import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from .evaluate_perception import (
        _load_oracle_map,
        _parse_oracle_id_args,
        _save_json,
        _write_csv,
        evaluate_result,
    )
except ImportError:
    from evaluate_perception import (
        _load_oracle_map,
        _parse_oracle_id_args,
        _save_json,
        _write_csv,
        evaluate_result,
    )


DEFAULT_GROUPS = ("task", "object", "text", "task,object")


def _default_dump_root() -> str:
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(src_dir, "perception_dumps")


def _default_out_dir() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(_default_dump_root(), "eval", f"benchmark_{stamp}")


def _discover_result_paths(
    roots: Iterable[str],
    glob_patterns: Iterable[str],
    explicit_summaries: Iterable[str],
    selected_only: bool = False,
) -> List[str]:
    import glob

    paths: List[str] = []
    for root in roots:
        root = os.path.abspath(root)
        if selected_only:
            paths.extend(_discover_selected_summary_paths(root))
        else:
            paths.extend(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True))
            paths.extend(glob.glob(os.path.join(root, "**", "instances.json"), recursive=True))
    for pattern in glob_patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    paths.extend(explicit_summaries)

    deduped = []
    seen = set()
    for path in paths:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return sorted(deduped)


def _discover_selected_summary_paths(root: str) -> List[str]:
    import glob

    selected_paths: List[str] = []
    for candidates_path in glob.glob(os.path.join(root, "**", "phrase_candidates.json"), recursive=True):
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        except Exception:
            continue
        for record in candidates.get("records") or []:
            if not record.get("selected"):
                continue
            summary_path = record.get("summary")
            if summary_path and os.path.exists(summary_path):
                selected_paths.append(summary_path)

    phrase_parent_dirs = set()
    for selected_path in selected_paths:
        phrase_parent_dirs.add(os.path.dirname(os.path.dirname(os.path.abspath(selected_path))))

    for summary_path in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
        abs_path = os.path.abspath(summary_path)
        parts = abs_path.replace(os.sep, "/").split("/")
        if any(part.startswith("phrase_") for part in parts):
            continue
        parent = os.path.dirname(abs_path)
        if parent in phrase_parent_dirs:
            continue
        selected_paths.append(abs_path)
    selected_paths.extend(glob.glob(os.path.join(root, "**", "instances.json"), recursive=True))
    return selected_paths


def _numeric_mean(rows: List[Dict[str, Any]], field: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _rate(count: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return float(count / total)


def _group_value(row: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    values = []
    for field in fields:
        value = row.get(field)
        values.append("unknown" if value in (None, "") else str(value))
    return " / ".join(values)


def _aggregate_rows(rows: List[Dict[str, Any]], fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_value(row, fields)].append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items()):
        total = len(group_rows)
        pass_count = sum(bool(row.get("pass")) for row in group_rows)
        oracle_rows = [row for row in group_rows if row.get("oracle_available")]
        oracle_pass_count = sum(bool(row.get("pass")) for row in oracle_rows)
        aggregate_rows.append(
            {
                "group_by": ",".join(fields),
                "group": key,
                "summary_count": total,
                "pass_count": int(pass_count),
                "pass_rate": _rate(pass_count, total),
                "oracle_count": len(oracle_rows),
                "oracle_pass_count": int(oracle_pass_count),
                "oracle_pass_rate": _rate(oracle_pass_count, len(oracle_rows)),
                "mean_center_dist_m": _numeric_mean(oracle_rows, "center_dist_m"),
                "mean_mask_iou": _numeric_mean(oracle_rows, "mean_mask_iou"),
                "mean_pred_points": _numeric_mean(group_rows, "pred_points"),
                "mean_used_cameras": _numeric_mean(group_rows, "used_cameras"),
                "mean_used_box_score": _numeric_mean(group_rows, "mean_used_box_score"),
            }
        )
    return aggregate_rows


def _write_aggregate_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "group_by",
        "group",
        "summary_count",
        "pass_count",
        "pass_rate",
        "oracle_count",
        "oracle_pass_count",
        "oracle_pass_rate",
        "mean_center_dist_m",
        "mean_mask_iou",
        "mean_pred_points",
        "mean_used_cameras",
        "mean_used_box_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _parse_group(value: str) -> Tuple[str, ...]:
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise ValueError("--group-by entries must include at least one field")
    return fields


def run_benchmark(args) -> Dict[str, Any]:
    summary_paths = _discover_result_paths(
        roots=args.root or [],
        glob_patterns=args.summaries_glob or [],
        explicit_summaries=args.summary or [],
        selected_only=getattr(args, "selected_only", False),
    )
    if args.max_summaries is not None:
        summary_paths = summary_paths[: args.max_summaries]
    if not summary_paths:
        raise ValueError("No summary.json or instances.json files found. Pass --root, --summary, or --summaries-glob.")

    oracle_map = _load_oracle_map(args.oracle_map)
    oracle_map.update(_parse_oracle_id_args(args.oracle_id))

    rows = []
    errors = []
    for summary_path in summary_paths:
        try:
            rows.append(
                evaluate_result(
                    result_path=summary_path,
                    object_name=args.object_name,
                    oracle_map=oracle_map,
                    center_pass_threshold=args.center_pass_threshold,
                    min_pred_points=args.min_pred_points,
                )
            )
        except Exception as exc:
            error_row = {
                "summary": summary_path,
                "error": str(exc),
            }
            errors.append(error_row)
            if not args.keep_going:
                raise

    if args.require_oracle:
        rows = [row for row in rows if row.get("oracle_available")]

    rows.sort(key=lambda row: (str(row.get("task")), str(row.get("object")), str(row.get("text")), str(row.get("summary"))))

    group_specs = [_parse_group(group) for group in (args.group_by or DEFAULT_GROUPS)]
    aggregate_rows = []
    for group_fields in group_specs:
        aggregate_rows.extend(_aggregate_rows(rows, group_fields))

    pass_count = sum(bool(row.get("pass")) for row in rows)
    oracle_rows = [row for row in rows if row.get("oracle_available")]
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_count": len(rows),
        "error_count": len(errors),
        "pass_count": int(pass_count),
        "pass_rate": _rate(pass_count, len(rows)),
        "oracle_count": len(oracle_rows),
        "oracle_pass_count": int(sum(bool(row.get("pass")) for row in oracle_rows)),
        "oracle_pass_rate": _rate(sum(bool(row.get("pass")) for row in oracle_rows), len(oracle_rows)),
        "mean_center_dist_m": _numeric_mean(oracle_rows, "center_dist_m"),
        "mean_mask_iou": _numeric_mean(oracle_rows, "mean_mask_iou"),
        "mean_pred_points": _numeric_mean(rows, "pred_points"),
        "mean_used_cameras": _numeric_mean(rows, "used_cameras"),
        "summary_paths": summary_paths,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "errors": errors,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch benchmark saved perception summary.json files. "
            "Use future dumps with manifest object_name_to_ids for automatic oracle metrics."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Root directory to recursively scan for summary.json. Defaults to src/perception_dumps.",
    )
    parser.add_argument("--summary", action="append", help="Explicit summary.json path. Can be repeated.")
    parser.add_argument("--summaries-glob", action="append", help="Glob for summary.json files.")
    parser.add_argument("--object", dest="object_name", help="Object name override for all summaries.")
    parser.add_argument(
        "--oracle-map",
        help='JSON string or path, e.g. {"rubbish":[12,13],"bin":[20]}.',
    )
    parser.add_argument("--oracle-id", action="append", help="Manual oracle id mapping, e.g. rubbish=12,13.")
    parser.add_argument("--center-pass-threshold", type=float, default=0.08)
    parser.add_argument("--min-pred-points", type=int, default=20)
    parser.add_argument("--require-oracle", action="store_true", help="Only report rows with oracle metrics.")
    parser.add_argument("--selected-only", action="store_true", help="Only evaluate final selected VLM phrase summaries.")
    parser.add_argument("--group-by", action="append", help="Comma-separated row fields, e.g. task,object.")
    parser.add_argument("--max-summaries", type=int, help="Limit number of summaries for a quick smoke run.")
    parser.add_argument("--keep-going", action="store_true", help="Record per-summary errors instead of stopping.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to perception_dumps/eval/benchmark_TIMESTAMP.")
    parser.add_argument("--no-write", action="store_true", help="Only print JSON to stdout.")
    args = parser.parse_args()

    if args.root is None and args.summary is None and args.summaries_glob is None:
        args.root = [_default_dump_root()]

    result = run_benchmark(args)

    if not args.no_write:
        out_dir = args.out_dir or _default_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        _save_json(os.path.join(out_dir, "benchmark.json"), result)
        _write_csv(os.path.join(out_dir, "details.csv"), result["rows"])
        _write_aggregate_csv(os.path.join(out_dir, "aggregate.csv"), result["aggregate_rows"])
        print(f"[benchmark] wrote {out_dir}")

    printable = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "aggregate_rows", "summary_paths"}
    }
    printable["aggregate_rows"] = result["aggregate_rows"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
