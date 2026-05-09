import argparse
import csv
import json
import os
import time
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any) -> str:
    value = _as_float(value)
    return "" if value is None else f"{value:.4f}"


def _format_vec(value: Any) -> str:
    if value is None:
        return ""
    try:
        return "[" + ", ".join(f"{float(item):.4f}" for item in value) + "]"
    except Exception:
        return str(value)


def _html_path(path: Optional[str], root: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, root)
        except ValueError:
            return path
    return path


def _rel(path: Optional[str], root: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _report_root(args) -> str:
    if args.out_dir:
        return os.path.abspath(args.out_dir)
    benchmark_dir = os.path.dirname(os.path.abspath(args.benchmark))
    return os.path.join(benchmark_dir, "failure_report")


def _is_selected_row(row: Dict[str, Any], args) -> bool:
    if args.include_passing:
        return True
    if not bool(row.get("pass")):
        return True
    center = _as_float(row.get("center_dist_m"))
    if args.center_threshold is not None and center is not None and center > args.center_threshold:
        return True
    iou = _as_float(row.get("mean_mask_iou"))
    if args.iou_threshold is not None and iou is not None and iou < args.iou_threshold:
        return True
    return False


def _severity(row: Dict[str, Any]) -> float:
    score = 0.0 if bool(row.get("pass")) else 100.0
    center = _as_float(row.get("center_dist_m"))
    if center is not None:
        score += center * 100.0
    iou = _as_float(row.get("mean_mask_iou"))
    if iou is not None:
        score += max(0.0, 1.0 - iou) * 10.0
    pred_points = _as_float(row.get("pred_points"))
    if pred_points is not None and pred_points < 20:
        score += 10.0
    return score


def _find_phrase_candidates(summary_path: str) -> Optional[str]:
    cur = os.path.abspath(os.path.dirname(summary_path))
    for _ in range(4):
        candidate = os.path.join(cur, "phrase_candidates.json")
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    return None


def _phrase_records(summary_path: str, report_dir: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    phrase_path = _find_phrase_candidates(summary_path)
    if not phrase_path:
        return None, []
    try:
        data = _load_json(phrase_path)
    except Exception:
        return phrase_path, []

    records = []
    for record in data.get("records") or []:
        score_info = record.get("score_info") or {}
        reasons = list(score_info.get("reasons") or [])
        if record.get("selection_reason"):
            reasons.insert(0, str(record.get("selection_reason")))
        if record.get("error"):
            reasons.append(str(record.get("error")))
        records.append(
            {
                "rank": record.get("rank"),
                "phrase": record.get("phrase"),
                "status": record.get("status"),
                "selected": bool(record.get("selected")),
                "score": score_info.get("score"),
                "fused_point_count": record.get("fused_point_count"),
                "fused_object_point_center": record.get("fused_object_point_center"),
                "summary": _rel(record.get("summary"), report_dir),
                "reasons": reasons,
            }
        )
    return phrase_path, records


def _camera_results(summary: Dict[str, Any], report_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for result in summary.get("camera_results") or []:
        fallback = result.get("candidate_fallback") or {}
        rows.append(
            {
                "camera": result.get("camera"),
                "accepted": bool(result.get("accepted")),
                "used_for_fusion": bool(result.get("used_for_fusion")),
                "reject_reason": result.get("reject_reason") or result.get("reason"),
                "box_score": result.get("box_score"),
                "mask_pixels": result.get("mask_pixels"),
                "object_point_count": result.get("object_point_count"),
                "object_point_center": result.get("object_point_center"),
                "label": result.get("label"),
                "selected_box_index": result.get("selected_box_index"),
                "overlay": _html_path(result.get("overlay"), report_dir),
                "rgb": _html_path(result.get("rgb"), report_dir),
                "fallback_selected": fallback.get("selected"),
                "fallback_score": fallback.get("selected_score"),
                "fallback_reasons": fallback.get("selected_reasons") or [],
                "candidate_count": len(result.get("candidates") or []),
            }
        )
    return rows


def _instance_diagnostics(
    summary: Dict[str, Any],
    selected_instance_id: Any,
    report_dir: str,
) -> Dict[str, Any]:
    candidates = []
    for index, candidate in enumerate(summary.get("candidates") or []):
        candidates.append(
            {
                "index": index,
                "camera": candidate.get("camera"),
                "accepted": bool(candidate.get("accepted")),
                "reject_reason": candidate.get("reject_reason"),
                "box_score": candidate.get("box_score"),
                "box_area_frac": candidate.get("box_area_frac"),
                "mask_pixels": candidate.get("mask_pixels"),
                "object_point_count": candidate.get("object_point_count"),
                "object_point_center": candidate.get("object_point_center"),
                "label": candidate.get("label"),
                "source_box_index": candidate.get("source_box_index"),
                "overlay": _html_path(candidate.get("overlay"), report_dir),
            }
        )

    instances = []
    selected_candidate_indices = set()
    for instance in summary.get("instances") or []:
        is_selected = str(instance.get("instance_id")) == str(selected_instance_id)
        if is_selected:
            selected_candidate_indices = {int(idx) for idx in instance.get("candidate_indices") or []}
        instances.append(
            {
                "instance_id": instance.get("instance_id"),
                "selected": is_selected,
                "fused_point_count": instance.get("fused_point_count"),
                "fused_object_point_center": instance.get("fused_object_point_center"),
                "candidate_count": instance.get("candidate_count"),
                "candidate_indices": instance.get("candidate_indices") or [],
                "cameras": instance.get("cameras") or [],
            }
        )

    for candidate in candidates:
        candidate["used_by_selected_instance"] = int(candidate["index"]) in selected_candidate_indices

    return {
        "candidate_count": summary.get("candidate_count"),
        "accepted_candidate_count": summary.get("accepted_candidate_count"),
        "instance_count": summary.get("instance_count"),
        "cluster_distance": summary.get("cluster_distance"),
        "instances": instances,
        "candidates": candidates,
    }


def _load_result_details(row: Dict[str, Any], report_dir: str) -> Dict[str, Any]:
    summary_path = row.get("summary")
    details: Dict[str, Any] = {
        "summary_exists": bool(summary_path and os.path.exists(summary_path)),
        "phrase_candidates": None,
        "phrases": [],
        "camera_results": [],
        "instances": [],
        "instance_candidates": [],
    }
    if not summary_path or not os.path.exists(summary_path):
        return details

    summary = _load_json(summary_path)
    phrase_path, phrases = _phrase_records(summary_path, report_dir)
    details["phrase_candidates"] = _rel(phrase_path, report_dir) if phrase_path else None
    details["phrases"] = phrases
    details["manifest"] = _rel(summary.get("manifest"), report_dir)
    details["grounding_model"] = summary.get("grounding_model")
    details["sam2_model"] = summary.get("sam2_model")
    details["filters"] = {
        "selection": summary.get("selection"),
        "box_threshold": summary.get("box_threshold"),
        "min_points": summary.get("min_points"),
        "max_points": summary.get("max_points"),
        "min_score": summary.get("min_score"),
        "max_center_z": summary.get("max_center_z"),
        "max_box_area_frac": summary.get("max_box_area_frac"),
    }

    if os.path.basename(summary_path) == "instances.json":
        instance_data = _instance_diagnostics(summary, row.get("instance_id"), report_dir)
        details["instances"] = instance_data["instances"]
        details["instance_candidates"] = instance_data["candidates"]
        details["instance_meta"] = {
            key: instance_data.get(key)
            for key in ("candidate_count", "accepted_candidate_count", "instance_count", "cluster_distance")
        }
    else:
        details["camera_results"] = _camera_results(summary, report_dir)
    return details


def _find_live_roots(paths: Iterable[str]) -> List[str]:
    roots = set()
    marker = f"{os.sep}benchmark_live{os.sep}"
    for path in paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if marker not in abs_path:
            continue
        prefix, rest = abs_path.split(marker, 1)
        parts = rest.split(os.sep)
        if parts:
            roots.add(os.path.join(prefix, "benchmark_live", parts[0]))
    return sorted(roots)


def _load_live_errors(summary_paths: Iterable[str]) -> List[Dict[str, Any]]:
    errors = []
    for root in _find_live_roots(summary_paths):
        path = os.path.join(root, "_benchmark", "run_records.json")
        if not os.path.exists(path):
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        for record in data.get("records") or []:
            if record.get("status") == "error":
                item = dict(record)
                item["run_records"] = path
                item["live_root"] = root
                errors.append(item)
    return errors


def _flatten_report_row(item: Dict[str, Any]) -> Dict[str, Any]:
    row = item["row"]
    return {
        "pass": row.get("pass"),
        "severity": item.get("severity"),
        "task": row.get("task"),
        "object": row.get("object"),
        "text": row.get("text"),
        "result_type": row.get("result_type"),
        "fail_reason": row.get("fail_reason"),
        "center_dist_m": row.get("center_dist_m"),
        "mean_mask_iou": row.get("mean_mask_iou"),
        "pred_points": row.get("pred_points"),
        "used_cameras": row.get("used_cameras"),
        "mean_used_box_score": row.get("mean_used_box_score"),
        "instance_id": row.get("instance_id"),
        "summary": row.get("summary"),
        "phrase_candidates": item.get("details", {}).get("phrase_candidates"),
        "selected_or_candidate_phrase_count": len(item.get("details", {}).get("phrases") or []),
    }


def _css() -> str:
    return """<style>
body { font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }
main { max-width: 1440px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 12px; font-size: 28px; }
h2 { margin-top: 28px; border-bottom: 1px solid #d7dce2; padding-bottom: 8px; }
h3 { margin: 0 0 10px; font-size: 16px; }
.summary { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 22px; }
.pill { display: inline-block; padding: 5px 9px; border-radius: 999px; background: #e8edf3; font-size: 13px; }
.bad { background: #ffe1df; color: #84221b; }
.warn { background: #fff1c7; color: #674a00; }
.good { background: #d9f2df; color: #145523; }
.case { margin: 18px 0; padding: 14px; background: white; border: 1px solid #ccd3db; border-radius: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.card { background: #fff; border: 1px solid #d7dce2; border-radius: 6px; padding: 10px; }
.card.used { border-color: #32a852; }
.card.rejected { border-color: #d75f4f; }
.card.selected { border-color: #2c77d4; }
.card img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #e3e6ea; background: #f0f2f5; }
.kv { margin: 4px 0; font-size: 13px; }
.kv b { display: inline-block; min-width: 120px; }
.muted { color: #637080; }
table { width: 100%; border-collapse: collapse; background: white; margin: 8px 0 12px; }
th, td { border-bottom: 1px solid #e1e5ea; padding: 7px; text-align: left; font-size: 13px; vertical-align: top; }
code { background: #eef1f4; padding: 1px 4px; border-radius: 3px; }
pre { overflow-x: auto; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 6px; }
</style>"""


def _kv(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = _format_vec(value)
    if value is None:
        value = ""
    return f"<div class='kv'><b>{escape(str(key))}</b> {escape(str(value))}</div>"


def _render_phrase_table(phrases: List[Dict[str, Any]]) -> str:
    if not phrases:
        return "<p class='muted'>No phrase candidate file found for this row.</p>"
    rows = [
        "<table><tr><th>rank</th><th>selected</th><th>phrase</th><th>status</th><th>score</th><th>points</th><th>center</th><th>reasons</th></tr>"
    ]
    for phrase in phrases:
        rows.append(
            "<tr>"
            f"<td>{escape(str(phrase.get('rank')))}</td>"
            f"<td>{'yes' if phrase.get('selected') else ''}</td>"
            f"<td>{escape(str(phrase.get('phrase') or ''))}</td>"
            f"<td>{escape(str(phrase.get('status') or ''))}</td>"
            f"<td>{escape(_format_float(phrase.get('score')))}</td>"
            f"<td>{escape(str(phrase.get('fused_point_count') or ''))}</td>"
            f"<td>{escape(_format_vec(phrase.get('fused_object_point_center')))}</td>"
            f"<td>{escape('; '.join(str(item) for item in phrase.get('reasons') or []))}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _render_camera_cards(cameras: List[Dict[str, Any]]) -> str:
    if not cameras:
        return ""
    rows = ["<h3>Camera Results</h3><div class='grid'>"]
    for result in cameras:
        used = bool(result.get("used_for_fusion"))
        card_class = "used" if used else "rejected"
        rows.append(f"<div class='card {card_class}'>")
        rows.append(f"<h3>{escape(str(result.get('camera')))}</h3>")
        if result.get("overlay"):
            rows.append(f"<img src='{escape(result.get('overlay'))}'>")
        rows.append(_kv("status", "used" if used else f"rejected: {result.get('reject_reason') or ''}"))
        rows.append(_kv("score", _format_float(result.get("box_score"))))
        rows.append(_kv("points", result.get("object_point_count")))
        rows.append(_kv("mask pixels", result.get("mask_pixels")))
        rows.append(_kv("center", result.get("object_point_center")))
        rows.append(_kv("label", result.get("label")))
        rows.append(_kv("fallback", ", ".join(result.get("fallback_reasons") or [])))
        rows.append("</div>")
    rows.append("</div>")
    return "\n".join(rows)


def _render_instance_section(item: Dict[str, Any]) -> str:
    details = item.get("details") or {}
    instances = details.get("instances") or []
    candidates = details.get("instance_candidates") or []
    if not instances and not candidates:
        return ""

    rows = ["<h3>Instances</h3>"]
    rows.append("<table><tr><th>id</th><th>selected</th><th>points</th><th>center</th><th>cameras</th><th>candidate indices</th></tr>")
    for instance in instances:
        rows.append(
            "<tr>"
            f"<td>{escape(str(instance.get('instance_id')))}</td>"
            f"<td>{'yes' if instance.get('selected') else ''}</td>"
            f"<td>{escape(str(instance.get('fused_point_count') or ''))}</td>"
            f"<td>{escape(_format_vec(instance.get('fused_object_point_center')))}</td>"
            f"<td>{escape(', '.join(instance.get('cameras') or []))}</td>"
            f"<td>{escape(', '.join(str(idx) for idx in instance.get('candidate_indices') or []))}</td>"
            "</tr>"
        )
    rows.append("</table><h3>Instance Candidates</h3><div class='grid'>")
    for candidate in candidates:
        card_class = "selected" if candidate.get("used_by_selected_instance") else ("used" if candidate.get("accepted") else "rejected")
        rows.append(f"<div class='card {card_class}'>")
        rows.append(
            f"<h3>{escape(str(candidate.get('camera')))} "
            f"idx {escape(str(candidate.get('index')))}</h3>"
        )
        if candidate.get("overlay"):
            rows.append(f"<img src='{escape(candidate.get('overlay'))}'>")
        rows.append(_kv("selected inst", candidate.get("used_by_selected_instance")))
        rows.append(_kv("accepted", candidate.get("accepted")))
        rows.append(_kv("score", _format_float(candidate.get("box_score"))))
        rows.append(_kv("area frac", _format_float(candidate.get("box_area_frac"))))
        rows.append(_kv("points", candidate.get("object_point_count")))
        rows.append(_kv("center", candidate.get("object_point_center")))
        rows.append(_kv("label", candidate.get("label")))
        rows.append("</div>")
    rows.append("</div>")
    return "\n".join(rows)


def _render_html(
    report: Dict[str, Any],
    cases: List[Dict[str, Any]],
    live_errors: List[Dict[str, Any]],
    out_path: str,
):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    rows = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>VoxPoser Perception Failure Report</title>",
        _css(),
        "</head><body><main>",
        "<h1>VoxPoser Perception Failure Report</h1>",
        "<section class='summary'>",
        f"<span class='pill'>created: {escape(str(report.get('created_at')))}</span>",
        f"<span class='pill'>selected rows: {len(cases)}</span>",
        f"<span class='pill'>live errors: {len(live_errors)}</span>",
        f"<span class='pill'>benchmark rows: {report.get('benchmark_summary', {}).get('summary_count')}</span>",
        "</section>",
        _kv("benchmark", _rel(report.get("benchmark"), out_dir)),
    ]

    if live_errors:
        rows.append("<h2>Live Detection Errors</h2>")
        rows.append("<table><tr><th>task</th><th>episode</th><th>object</th><th>error</th><th>run_records</th></tr>")
        for error in live_errors:
            rows.append(
                "<tr>"
                f"<td>{escape(str(error.get('task')))}</td>"
                f"<td>{escape(str(error.get('episode')))}</td>"
                f"<td>{escape(str(error.get('object')))}</td>"
                f"<td>{escape(str(error.get('error')))}</td>"
                f"<td><code>{escape(_rel(error.get('run_records'), out_dir))}</code></td>"
                "</tr>"
            )
        rows.append("</table>")

    rows.append("<h2>Selected Benchmark Rows</h2>")
    for item in cases:
        row = item["row"]
        status_class = "bad" if not row.get("pass") else "warn"
        rows.append("<section class='case'>")
        rows.append(
            f"<h3>{escape(str(row.get('task')))} / {escape(str(row.get('object')))} "
            f"/ {escape(str(row.get('text')))}</h3>"
        )
        rows.append("<section class='summary'>")
        rows.append(f"<span class='pill {status_class}'>pass: {escape(str(row.get('pass')))}</span>")
        rows.append(f"<span class='pill'>reason: {escape(str(row.get('fail_reason') or ''))}</span>")
        rows.append(f"<span class='pill'>center: {escape(_format_float(row.get('center_dist_m')))}m</span>")
        rows.append(f"<span class='pill'>IoU: {escape(_format_float(row.get('mean_mask_iou')))}</span>")
        rows.append(f"<span class='pill'>points: {escape(str(row.get('pred_points')))}</span>")
        rows.append(f"<span class='pill'>type: {escape(str(row.get('result_type')))}</span>")
        rows.append("</section>")
        rows.append(_kv("summary", _rel(row.get("summary"), out_dir)))
        rows.append(_kv("pred center", row.get("pred_center")))
        rows.append(_kv("oracle center", row.get("oracle_center")))
        rows.append(_kv("oracle ids", row.get("oracle_ids")))
        rows.append("<h3>Phrase Candidates</h3>")
        rows.append(_render_phrase_table(item.get("details", {}).get("phrases") or []))
        rows.append(_render_camera_cards(item.get("details", {}).get("camera_results") or []))
        rows.append(_render_instance_section(item))
        rows.append("</section>")

    rows.extend(["</main></body></html>"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def generate_report(args) -> Dict[str, Any]:
    benchmark = _load_json(args.benchmark)
    report_dir = _report_root(args)
    os.makedirs(report_dir, exist_ok=True)

    rows = [row for row in benchmark.get("rows") or [] if _is_selected_row(row, args)]
    rows.sort(key=_severity, reverse=True)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    cases = []
    for row in rows:
        cases.append(
            {
                "severity": _severity(row),
                "row": row,
                "details": _load_result_details(row, report_dir),
            }
        )

    live_errors = _load_live_errors(row.get("summary") for row in benchmark.get("rows") or [])
    if args.max_live_errors is not None:
        live_errors = live_errors[: args.max_live_errors]

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": os.path.abspath(args.benchmark),
        "benchmark_summary": {
            "summary_count": benchmark.get("summary_count"),
            "pass_count": benchmark.get("pass_count"),
            "pass_rate": benchmark.get("pass_rate"),
            "oracle_count": benchmark.get("oracle_count"),
            "mean_center_dist_m": benchmark.get("mean_center_dist_m"),
            "mean_mask_iou": benchmark.get("mean_mask_iou"),
            "error_count": benchmark.get("error_count"),
        },
        "filters": {
            "include_passing": args.include_passing,
            "center_threshold": args.center_threshold,
            "iou_threshold": args.iou_threshold,
            "max_rows": args.max_rows,
        },
        "selected_count": len(cases),
        "live_error_count": len(live_errors),
        "live_errors": live_errors,
        "cases": cases,
    }

    _save_json(os.path.join(report_dir, "failure_report.json"), report)
    _write_csv(
        os.path.join(report_dir, "failures.csv"),
        [_flatten_report_row(item) for item in cases],
        [
            "pass",
            "severity",
            "task",
            "object",
            "text",
            "result_type",
            "fail_reason",
            "center_dist_m",
            "mean_mask_iou",
            "pred_points",
            "used_cameras",
            "mean_used_box_score",
            "instance_id",
            "summary",
            "phrase_candidates",
            "selected_or_candidate_phrase_count",
        ],
    )
    if live_errors:
        _write_csv(
            os.path.join(report_dir, "live_errors.csv"),
            live_errors,
            ["task", "episode", "object", "instruction", "status", "started_at", "error", "run_records", "live_root"],
        )
    _render_html(report, cases, live_errors, os.path.join(report_dir, "index.html"))
    return {
        "out_dir": report_dir,
        "html": os.path.join(report_dir, "index.html"),
        "json": os.path.join(report_dir, "failure_report.json"),
        "csv": os.path.join(report_dir, "failures.csv"),
        "selected_count": len(cases),
        "live_error_count": len(live_errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a browsable failure report from a perception benchmark.json."
    )
    parser.add_argument("--benchmark", required=True, help="Path to benchmark.json.")
    parser.add_argument("--out-dir", help="Output directory. Defaults to BENCHMARK_DIR/failure_report.")
    parser.add_argument("--include-passing", action="store_true", help="Include every benchmark row.")
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=None,
        help="Also include passing rows with center_dist_m above this value.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Also include passing rows with mean_mask_iou below this value.",
    )
    parser.add_argument("--max-rows", type=int, help="Limit selected benchmark rows.")
    parser.add_argument("--max-live-errors", type=int, help="Limit live run error rows.")
    args = parser.parse_args()

    result = generate_report(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
