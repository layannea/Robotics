import json
import os
import re
import subprocess
import time
from html import escape
from typing import Dict, Iterable, List, Optional

import numpy as np

from .dump_rlbench_obs import dump_current_obs
from .role_aware import object_entry_for_query, role_for_query
from .scorer_v2 import score_camera_candidate_v2, score_phrase_summary_v2


class SubprocessPerceptionBackend:
    """Ground objects with the Python 3.11 perception environment.

    RLBench/PyRep stays in the original Python 3.9 environment, while
    GroundingDINO/SAM2 runs in a separate conda env. This subprocess bridge is
    deliberately simple and slow; it is the safest integration point before
    moving to a persistent perception server.
    """

    DEFAULT_CAMERAS = ("front", "overhead", "wrist")

    SINGLE_OBJECT_CONFIGS = {
        "button": {
            "text": "red button",
            "max_center_z": 1.0,
            "max_points": 1000,
            "min_points": 20,
            "selection": "smallest_labeled_area",
        },
        "bin": {
            "text": "black bin",
            "max_center_z": 1.2,
            "max_points": 5000,
            "min_points": 20,
            "selection": "smallest_labeled_area",
        },
        "rubbish": {
            "text": "gray trash",
            "max_center_z": 1.2,
            "max_points": 2000,
            "min_points": 20,
            "selection": "smallest_labeled_area",
        },
    }

    INSTANCE_OBJECT_CONFIGS = {
        "tomato": {
            "text": "red tomato",
            "max_center_z": 1.2,
            "max_points": 1000,
            "min_points": 10,
            "max_box_area_frac": 0.08,
            "cluster_distance": 0.08,
            "min_instance_cameras": 2,
        }
    }

    def __init__(
        self,
        src_dir: str,
        conda_env: str = "voxposer-perception",
        cameras: Iterable[str] = DEFAULT_CAMERAS,
        dump_root: Optional[str] = None,
        grounding_scale: float = 4.0,
        box_threshold: float = 0.15,
        min_score: float = 0.15,
        timeout: int = 180,
        use_openai_vlm: bool = False,
        vlm_model: str = "gpt-5.4",
        vlm_instruction: Optional[str] = None,
        vlm_object_context: str = "",
        vlm_phrase_count: int = 4,
        vlm_panel_size: int = 384,
        vlm_detail: str = "high",
        vlm_api_mode: str = "responses",
        vlm_api_key_env: str = "OPENAI_API_KEY",
        vlm_base_url: Optional[str] = None,
        vlm_reasoning_effort: str = "low",
        vlm_use_candidate_scorer: bool = True,
        vlm_max_phrases: int = 2,
        vlm_early_stop_score: float = 2.0,
    ):
        self.src_dir = os.path.abspath(src_dir)
        self.conda_env = conda_env
        self.cameras = tuple(cameras)
        self.dump_root = dump_root or os.path.join(self.src_dir, "perception_dumps", "live")
        self.grounding_scale = grounding_scale
        self.box_threshold = box_threshold
        self.min_score = min_score
        self.timeout = timeout
        self._cache: Dict[tuple, np.ndarray] = {}
        self.use_openai_vlm = use_openai_vlm
        self.vlm_model = vlm_model
        self.vlm_instruction = vlm_instruction
        self.vlm_object_context = vlm_object_context
        self.vlm_phrase_count = vlm_phrase_count
        self.vlm_panel_size = vlm_panel_size
        self.vlm_detail = vlm_detail
        self.vlm_api_mode = vlm_api_mode
        self.vlm_api_key_env = vlm_api_key_env
        self.vlm_base_url = vlm_base_url
        self.vlm_reasoning_effort = vlm_reasoning_effort
        self.vlm_use_candidate_scorer = vlm_use_candidate_scorer
        self.vlm_max_phrases = vlm_max_phrases
        self.vlm_early_stop_score = vlm_early_stop_score
        self._vlm_cache: Dict[tuple, Dict] = {}

    def clear_cache(self):
        self._cache.clear()
        self._vlm_cache.clear()

    def set_vlm_instruction(self, instruction: Optional[str]):
        self.vlm_instruction = instruction
        self._vlm_cache.clear()

    def get_object_points(self, env, query_name: str) -> np.ndarray:
        if env.latest_obs is None:
            raise ValueError("env.latest_obs is None. Call env.reset() first.")

        cache_key = (id(env.latest_obs), query_name)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        dump_dir = self._make_dump_dir(env, query_name)
        manifest_path = dump_current_obs(env, out_dir=dump_dir, cameras=self.cameras)
        vlm_descriptions = None
        if self.use_openai_vlm:
            vlm_descriptions = self._get_openai_vlm_descriptions(env, manifest_path, dump_dir)

        try:
            points = self._run_query(query_name, manifest_path, dump_dir, vlm_descriptions)
            if len(points) == 0:
                raise ValueError(f"Perception backend found no points for {query_name!r}")
        except Exception as e:
            self._write_perception_diagnostics(
                dump_dir=dump_dir,
                query_name=query_name,
                manifest_path=manifest_path,
                vlm_descriptions=vlm_descriptions,
                error=str(e),
            )
            raise
        self._write_perception_diagnostics(
            dump_dir=dump_dir,
            query_name=query_name,
            manifest_path=manifest_path,
            vlm_descriptions=vlm_descriptions,
            final_point_count=len(points),
        )
        self._cache[cache_key] = points
        return points.copy()

    def _make_dump_dir(self, env, query_name: str) -> str:
        task_name = env.task.get_name() if env.task is not None else "unknown_task"
        safe_query = re.sub(r"[^a-zA-Z0-9_.-]+", "_", query_name)
        stamp = f"{int(time.time() * 1000)}_{id(env.latest_obs)}"
        return os.path.join(self.dump_root, task_name, f"{safe_query}_{stamp}")

    def _run_query(
        self,
        query_name: str,
        manifest_path: str,
        dump_dir: str,
        vlm_descriptions: Optional[Dict] = None,
    ) -> np.ndarray:
        query_lower = query_name.lower()
        vlm_phrases = self._phrases_for_query(query_name, vlm_descriptions)
        tomato_match = re.fullmatch(r"tomato(\d+)?", query_lower)
        if tomato_match:
            instance_number = int(tomato_match.group(1) or "1")
            if vlm_phrases:
                instance_config = self.INSTANCE_OBJECT_CONFIGS["tomato"].copy()
                instance_config["text"] = vlm_phrases[0]
                instance_config.update(self._role_for_query(query_name, vlm_descriptions))
                return self._run_instance_query(
                    base_name="tomato",
                    instance_index=max(0, instance_number - 1),
                    manifest_path=manifest_path,
                    out_dir=os.path.join(dump_dir, "perception_tomato"),
                    config_override=instance_config,
                )
            return self._run_instance_query(
                base_name="tomato",
                instance_index=max(0, instance_number - 1),
                manifest_path=manifest_path,
                out_dir=os.path.join(dump_dir, "perception_tomato"),
            )

        config = self.SINGLE_OBJECT_CONFIGS.get(query_lower, {"text": query_name})
        if vlm_phrases:
            config = config.copy()
            config.update(self._role_for_query(query_name, vlm_descriptions))
            return self._run_single_query_phrases(
                query_name=query_name,
                config=config,
                phrases=vlm_phrases,
                manifest_path=manifest_path,
                out_dir=os.path.join(dump_dir, f"perception_{query_lower}"),
            )
        return self._run_single_query(
            config=config,
            manifest_path=manifest_path,
            out_dir=os.path.join(dump_dir, f"perception_{query_lower}"),
        )

    def _phrases_for_query(self, query_name: str, vlm_descriptions: Optional[Dict]) -> List[str]:
        if not vlm_descriptions:
            return []
        entry = object_entry_for_query(query_name, vlm_descriptions)
        if not isinstance(entry, dict):
            return []

        phrases = []
        best_phrase = entry.get("best_phrase") or entry.get("grounding_text")
        if best_phrase:
            phrases.append(best_phrase)
        phrases.extend(entry.get("phrases") or [])

        deduped = []
        for phrase in phrases:
            phrase = str(phrase).strip()
            if phrase and phrase not in deduped:
                deduped.append(phrase)
        return deduped

    def _role_for_query(self, query_name: str, vlm_descriptions: Optional[Dict]) -> Dict:
        return role_for_query(query_name, vlm_descriptions)

    def _get_openai_vlm_descriptions(self, env, manifest_path: str, dump_dir: str) -> Dict:
        instruction = self.vlm_instruction or ""
        if not instruction:
            task_name = env.task.get_name() if env.task is not None else "unknown task"
            instruction = f"Describe task-relevant objects for the RLBench task {task_name}."
        object_names = env.get_object_names()
        cache_key = (
            id(env.latest_obs),
            instruction,
            tuple(object_names),
            self.vlm_object_context,
            self.vlm_model,
            self.vlm_phrase_count,
            self.vlm_panel_size,
        )
        if cache_key in self._vlm_cache:
            return self._vlm_cache[cache_key]

        out_dir = os.path.join(dump_dir, "openai_vlm")
        summary_path = os.path.join(out_dir, "vlm_descriptions.json")
        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            os.path.join(self.src_dir, "perception", "run_openai_vlm_describer.py"),
            "--manifest",
            manifest_path,
            "--instruction",
            instruction,
            "--objects",
            ",".join(object_names),
            "--out-dir",
            out_dir,
            "--model",
            self.vlm_model,
            "--panel-size",
            str(self.vlm_panel_size),
            "--phrase-count",
            str(self.vlm_phrase_count),
            "--detail",
            self.vlm_detail,
            "--api-mode",
            self.vlm_api_mode,
            "--api-key-env",
            self.vlm_api_key_env,
            "--reasoning-effort",
            self.vlm_reasoning_effort,
        ]
        if self.vlm_object_context:
            cmd.extend(["--object-context", self.vlm_object_context])
        if self.vlm_base_url:
            cmd.extend(["--base-url", self.vlm_base_url])

        self._run_command(cmd)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self._vlm_cache[cache_key] = summary
        return summary

    def _run_single_query_phrases(
        self,
        query_name: str,
        config: Dict,
        phrases: List[str],
        manifest_path: str,
        out_dir: str,
    ) -> np.ndarray:
        errors = []
        candidates = []
        phrase_records = []
        phrases_to_try = phrases[: self.vlm_max_phrases] if self.vlm_max_phrases else phrases
        for idx, phrase in enumerate(phrases_to_try):
            phrase_config = config.copy()
            phrase_config["text"] = phrase
            phrase_out_dir = os.path.join(out_dir, f"phrase_{idx:02d}_{self._safe_name(phrase)}")
            phrase_record = {
                "rank": idx,
                "phrase": phrase,
                "out_dir": phrase_out_dir,
                "summary": os.path.join(phrase_out_dir, "summary.json"),
                "status": "started",
            }
            try:
                points, summary = self._run_single_query_with_summary(
                    config=phrase_config,
                    manifest_path=manifest_path,
                    out_dir=phrase_out_dir,
                )
                phrase_record["fused_point_count"] = int(summary.get("fused_point_count") or len(points))
                phrase_record["fused_object_point_center"] = summary.get("fused_object_point_center")
                if len(points) > 0:
                    if not self.vlm_use_candidate_scorer:
                        phrase_record["status"] = "accepted"
                        phrase_record["selected"] = True
                        phrase_record["selection_reason"] = "first_non_empty"
                        phrase_records.append(phrase_record)
                        self._write_phrase_attempts(out_dir, phrase_records)
                        print(f"[perception] VLM phrase accepted: {phrase!r}")
                        return points

                    score_info = self._score_single_query_candidate(
                        query_name=query_name,
                        phrase=phrase,
                        phrase_rank=idx,
                        config=phrase_config,
                        summary=summary,
                        points=points,
                    )
                    phrase_record["status"] = "scored"
                    phrase_record["score_info"] = score_info
                    candidates.append((score_info["score"], phrase, points, score_info))
                    print(
                        "[perception] VLM candidate "
                        f"phrase={phrase!r} score={score_info['score']:.3f} "
                        f"reasons={score_info['reasons']}"
                    )
                    if score_info["score"] >= self.vlm_early_stop_score:
                        phrase_record["selected"] = True
                        phrase_record["selection_reason"] = "early_stop"
                        phrase_records.append(phrase_record)
                        self._write_phrase_attempts(out_dir, phrase_records)
                        print(f"[perception] VLM phrase early accepted: {phrase!r}")
                        return points
                else:
                    phrase_record["status"] = "empty_points"
                errors.append(f"{phrase!r}: empty points")
            except Exception as e:
                phrase_record["status"] = "error"
                phrase_record["error"] = str(e)
                errors.append(f"{phrase!r}: {e}")
            phrase_records.append(phrase_record)

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_phrase, best_points, best_info = candidates[0]
            for phrase_record in phrase_records:
                if phrase_record.get("phrase") == best_phrase:
                    phrase_record["selected"] = True
                    phrase_record["selection_reason"] = "best_score"
                    break
            self._write_phrase_attempts(out_dir, phrase_records)
            print(
                "[perception] VLM phrase accepted by scorer: "
                f"{best_phrase!r} score={best_score:.3f} reasons={best_info['reasons']}"
            )
            return best_points
        self._write_phrase_attempts(out_dir, phrase_records)
        raise ValueError("All VLM phrases failed:\n" + "\n".join(errors))

    def _run_single_query(self, config: Dict, manifest_path: str, out_dir: str) -> np.ndarray:
        points, _ = self._run_single_query_with_summary(config, manifest_path, out_dir)
        return points

    def _run_single_query_with_summary(
        self,
        config: Dict,
        manifest_path: str,
        out_dir: str,
    ) -> tuple:
        summary_path = os.path.join(out_dir, "summary.json")
        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            os.path.join(self.src_dir, "perception", "run_multicam_grounded_sam2.py"),
            "--manifest",
            manifest_path,
            "--text",
            config["text"],
            "--out-dir",
            out_dir,
            "--cameras",
            *self.cameras,
            "--grounding-scale",
            str(self.grounding_scale),
            "--box-threshold",
            str(self.box_threshold),
            "--selection",
            config.get("selection", "smallest_labeled_area"),
            "--min-points",
            str(config.get("min_points", 20)),
            "--min-score",
            str(self.min_score),
        ]
        if config.get("max_points") is not None:
            cmd.extend(["--max-points", str(config["max_points"])])
        if config.get("max_center_z") is not None:
            cmd.extend(["--max-center-z", str(config["max_center_z"])])

        self._run_command(cmd)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        points_path = summary["fused_object_points"]
        return np.load(points_path), summary

    def _score_single_query_candidate(
        self,
        query_name: str,
        phrase: str,
        phrase_rank: int,
        config: Dict,
        summary: Dict,
        points: np.ndarray,
    ) -> Dict:
        return score_phrase_summary_v2(
            query_name=query_name,
            phrase=phrase,
            phrase_rank=phrase_rank,
            config=config,
            summary=summary,
            points=points,
        )

    def _preferred_center_z_max(self, summary: Dict, query_name: str) -> Optional[float]:
        manifest_path = summary.get("manifest")
        table_z = None
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                bounds_min = manifest.get("workspace_bounds_min")
                if bounds_min is not None:
                    table_z = float(bounds_min[2])
            except Exception:
                table_z = None

        query_lower = query_name.lower()
        if table_z is not None:
            if query_lower in {"rubbish", "button"} or re.fullmatch(r"tomato\d*", query_lower):
                return table_z + 0.22
            if query_lower == "bin":
                return table_z + 0.35
        if query_lower in {"rubbish", "button"} or re.fullmatch(r"tomato\d*", query_lower):
            return 0.98
        if query_lower == "bin":
            return 1.10
        return None

    def _run_instance_query(
        self,
        base_name: str,
        instance_index: int,
        manifest_path: str,
        out_dir: str,
        config_override: Optional[Dict] = None,
    ) -> np.ndarray:
        config = config_override or self.INSTANCE_OBJECT_CONFIGS[base_name]
        summary_path = os.path.join(out_dir, "instances.json")
        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            os.path.join(self.src_dir, "perception", "run_instance_grounded_sam2.py"),
            "--manifest",
            manifest_path,
            "--text",
            config["text"],
            "--out-dir",
            out_dir,
            "--cameras",
            *self.cameras,
            "--grounding-scale",
            str(self.grounding_scale),
            "--box-threshold",
            str(self.box_threshold),
            "--max-candidates-per-camera",
            "5",
            "--min-points",
            str(config.get("min_points", 10)),
            "--min-score",
            str(self.min_score),
            "--cluster-distance",
            str(config.get("cluster_distance", 0.08)),
            "--max-box-area-frac",
            str(config.get("max_box_area_frac", 0.08)),
        ]
        if config.get("max_points") is not None:
            cmd.extend(["--max-points", str(config["max_points"])])
        if config.get("max_center_z") is not None:
            cmd.extend(["--max-center-z", str(config["max_center_z"])])

        self._run_command(cmd)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        instances = sorted(
            [
                instance
                for instance in summary["instances"]
                if len(instance.get("cameras", [])) >= config.get("min_instance_cameras", 1)
            ],
            key=lambda inst: inst["fused_object_point_center"][1],
        )
        if instance_index >= len(instances):
            raise ValueError(
                f"Requested {base_name}{instance_index + 1}, but only found {len(instances)} instances"
            )
        return np.load(instances[instance_index]["fused_object_points"])

    def _run_command(self, cmd):
        os.makedirs(cmd[cmd.index("--out-dir") + 1], exist_ok=True)
        print(f"[perception] running: {' '.join(cmd)}")
        completed = subprocess.run(
            cmd,
            cwd=self.src_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Perception subprocess failed with output:\n"
                f"{completed.stdout}"
            )
        print(completed.stdout)

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:80] or "phrase"

    def _write_perception_diagnostics(
        self,
        dump_dir: str,
        query_name: str,
        manifest_path: str,
        vlm_descriptions: Optional[Dict] = None,
        final_point_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        os.makedirs(dump_dir, exist_ok=True)
        diagnostics = {
            "query_name": query_name,
            "dump_dir": dump_dir,
            "manifest": manifest_path,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error" if error else "ok",
            "error": error,
            "final_point_count": final_point_count,
            "vlm": self._summarize_vlm_diagnostics(vlm_descriptions, dump_dir),
            "attempts": self._collect_perception_attempts(dump_dir),
        }

        json_path = os.path.join(dump_dir, "diagnostics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        html_path = os.path.join(dump_dir, "diagnostics.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._render_diagnostics_html(diagnostics, dump_dir))
        print(f"[perception] diagnostics: {html_path}")
        return html_path

    def _summarize_vlm_diagnostics(
        self,
        vlm_descriptions: Optional[Dict],
        dump_dir: str,
    ) -> Optional[Dict]:
        summary = vlm_descriptions
        if summary is None:
            vlm_path = os.path.join(dump_dir, "openai_vlm", "vlm_descriptions.json")
            if os.path.exists(vlm_path):
                try:
                    with open(vlm_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    summary = None
        if not summary:
            return None

        parsed_objects = (summary.get("parsed_output") or {}).get("objects") or {}
        objects = {}
        for name, entry in parsed_objects.items():
            if not isinstance(entry, dict):
                continue
            objects[name] = {
                "role": entry.get("role"),
                "visible": entry.get("visible"),
                "best_phrase": entry.get("best_phrase") or entry.get("grounding_text"),
                "phrases": entry.get("phrases") or [],
                "reason": entry.get("reason"),
            }
        return {
            "model": summary.get("model"),
            "instruction": summary.get("instruction"),
            "objects": objects,
            "target_objects": summary.get("objects") or [],
            "cameras": summary.get("cameras") or [],
            "montage": summary.get("montage"),
            "summary": summary.get("summary"),
            "raw_output": summary.get("raw_output"),
            "usage": summary.get("usage"),
        }

    def _collect_perception_attempts(self, dump_dir: str) -> List[Dict]:
        attempts = []
        for root, _, files in os.walk(dump_dir):
            if "phrase_candidates.json" in files:
                path = os.path.join(root, "phrase_candidates.json")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    summary["type"] = "phrases"
                    summary["path"] = path
                    summary["relative_path"] = self._relative_path(path, dump_dir)
                    attempts.append(summary)
                except Exception as e:
                    attempts.append({"type": "phrases_error", "path": path, "error": str(e)})
            if "summary.json" in files:
                path = os.path.join(root, "summary.json")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    attempts.append(self._summarize_single_attempt(path, summary, dump_dir))
                except Exception as e:
                    attempts.append({"type": "single_error", "path": path, "error": str(e)})
            if "instances.json" in files:
                path = os.path.join(root, "instances.json")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    attempts.append(self._summarize_instance_attempt(path, summary, dump_dir))
                except Exception as e:
                    attempts.append({"type": "instance_error", "path": path, "error": str(e)})
        attempts.sort(key=lambda item: item.get("path", ""))
        return attempts

    def _write_phrase_attempts(self, out_dir: str, phrase_records: List[Dict]) -> str:
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            "type": "phrases",
            "phrase_count": len(phrase_records),
            "selected_phrase": next(
                (record.get("phrase") for record in phrase_records if record.get("selected")),
                None,
            ),
            "records": phrase_records,
        }
        path = os.path.join(out_dir, "phrase_candidates.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return path

    def _summarize_single_attempt(self, path: str, summary: Dict, dump_dir: str) -> Dict:
        camera_results = []
        for result in summary.get("camera_results", []):
            camera_results.append(
                {
                    "camera": result.get("camera"),
                    "accepted": result.get("accepted"),
                    "used_for_fusion": result.get("used_for_fusion"),
                    "reject_reason": result.get("reject_reason") or result.get("reason"),
                    "box_score": result.get("box_score"),
                    "mask_pixels": result.get("mask_pixels"),
                    "object_point_count": result.get("object_point_count"),
                    "object_point_center": result.get("object_point_center"),
                    "overlay": result.get("overlay"),
                    "rgb": result.get("rgb"),
                    "selected_box_index": result.get("selected_box_index"),
                    "box": result.get("box"),
                    "label": result.get("label"),
                    "candidate_count": len(result.get("candidates") or []),
                    "candidates": result.get("candidates") or [],
                    "candidate_fallback": result.get("candidate_fallback"),
                }
            )
        return {
            "type": "single",
            "path": path,
            "relative_path": self._relative_path(path, dump_dir),
            "text": summary.get("text"),
            "grounding_model": summary.get("grounding_model"),
            "sam2_model": summary.get("sam2_model"),
            "fused_point_count": summary.get("fused_point_count"),
            "fused_object_point_center": summary.get("fused_object_point_center"),
            "selection": summary.get("selection"),
            "min_points": summary.get("min_points"),
            "max_points": summary.get("max_points"),
            "min_score": summary.get("min_score"),
            "max_center_z": summary.get("max_center_z"),
            "camera_results": camera_results,
        }

    def _summarize_instance_attempt(self, path: str, summary: Dict, dump_dir: str) -> Dict:
        candidates = []
        for candidate in summary.get("candidates", []):
            candidates.append(
                {
                    "camera": candidate.get("camera"),
                    "accepted": candidate.get("accepted"),
                    "reject_reason": candidate.get("reject_reason"),
                    "box_score": candidate.get("box_score"),
                    "box_area_frac": candidate.get("box_area_frac"),
                    "mask_pixels": candidate.get("mask_pixels"),
                    "object_point_count": candidate.get("object_point_count"),
                    "object_point_center": candidate.get("object_point_center"),
                    "overlay": candidate.get("overlay"),
                    "label": candidate.get("label"),
                    "source_box_index": candidate.get("source_box_index"),
                }
            )
        instances = []
        for instance in summary.get("instances", []):
            instances.append(
                {
                    "instance_id": instance.get("instance_id"),
                    "fused_point_count": instance.get("fused_point_count"),
                    "fused_object_point_center": instance.get("fused_object_point_center"),
                    "candidate_count": instance.get("candidate_count"),
                    "cameras": instance.get("cameras") or [],
                    "members": instance.get("members") or [],
                    "fused_object_points": instance.get("fused_object_points"),
                }
            )
        return {
            "type": "instance",
            "path": path,
            "relative_path": self._relative_path(path, dump_dir),
            "text": summary.get("text"),
            "grounding_model": summary.get("grounding_model"),
            "sam2_model": summary.get("sam2_model"),
            "candidate_count": summary.get("candidate_count"),
            "accepted_candidate_count": summary.get("accepted_candidate_count"),
            "instance_count": summary.get("instance_count"),
            "cluster_distance": summary.get("cluster_distance"),
            "max_box_area_frac": summary.get("max_box_area_frac"),
            "candidates": candidates,
            "instances": instances,
        }

    def _render_diagnostics_html(self, diagnostics: Dict, dump_dir: str) -> str:
        status = diagnostics.get("status")
        status_class = "bad" if status == "error" else "good"
        rows = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>VoxPoser Perception Diagnostics</title>",
            self._diagnostics_css(),
            "</head><body>",
            "<main>",
            f"<h1>Perception Diagnostics: {escape(str(diagnostics.get('query_name')))}</h1>",
            "<section class='summary'>",
            f"<span class='pill {status_class}'>status: {escape(str(status))}</span>",
            f"<span class='pill'>final points: {escape(str(diagnostics.get('final_point_count')))}</span>",
            f"<span class='pill'>attempts: {len(diagnostics.get('attempts') or [])}</span>",
            "</section>",
            self._render_key_value("dump", diagnostics.get("dump_dir")),
            self._render_key_value("manifest", diagnostics.get("manifest")),
        ]
        if diagnostics.get("error"):
            rows.append(f"<pre class='error'>{escape(str(diagnostics['error']))}</pre>")
        rows.append(self._render_vlm_section(diagnostics.get("vlm"), dump_dir))
        rows.append("<h2>Grounding Attempts</h2>")
        attempts = diagnostics.get("attempts") or []
        if not attempts:
            rows.append("<p>No GroundingDINO/SAM2 summaries were written.</p>")
        for attempt in attempts:
            if attempt.get("type") == "phrases":
                rows.append(self._render_phrase_attempts(attempt, dump_dir))
            elif attempt.get("type") == "single":
                rows.append(self._render_single_attempt(attempt, dump_dir))
            elif attempt.get("type") == "instance":
                rows.append(self._render_instance_attempt(attempt, dump_dir))
            else:
                rows.append(f"<pre class='error'>{escape(json.dumps(attempt, indent=2))}</pre>")
        rows.extend(["</main>", "</body></html>"])
        return "\n".join(rows)

    def _diagnostics_css(self) -> str:
        return """<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }
main { max-width: 1320px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 12px; font-size: 28px; }
h2 { margin-top: 28px; border-bottom: 1px solid #d7dce2; padding-bottom: 8px; }
h3 { margin: 0 0 10px; }
.summary { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.pill { display: inline-block; padding: 5px 9px; border-radius: 999px; background: #e8edf3; font-size: 13px; }
.good { background: #d9f2df; color: #145523; }
.bad { background: #ffe1df; color: #84221b; }
.used { border-color: #32a852; }
.rejected { border-color: #d75f4f; }
.muted { color: #637080; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.card { background: white; border: 1px solid #d7dce2; border-radius: 6px; padding: 12px; }
.card img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #e3e6ea; background: #f0f2f5; }
.kv { margin: 5px 0; font-size: 14px; }
.kv b { display: inline-block; min-width: 140px; }
.attempt { margin: 16px 0; padding: 14px; background: white; border: 1px solid #ccd3db; border-radius: 8px; }
pre { overflow-x: auto; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 6px; }
pre.error { background: #421b1b; }
table { width: 100%; border-collapse: collapse; background: white; }
th, td { border-bottom: 1px solid #e1e5ea; padding: 7px; text-align: left; font-size: 13px; vertical-align: top; }
code { background: #eef1f4; padding: 1px 4px; border-radius: 3px; }
</style>"""

    def _render_vlm_section(self, vlm: Optional[Dict], dump_dir: str) -> str:
        if not vlm:
            return "<h2>VLM</h2><p class='muted'>No VLM summary for this perception call.</p>"
        rows = ["<h2>VLM</h2>", "<section class='card'>"]
        rows.append(self._render_key_value("model", vlm.get("model")))
        rows.append(self._render_key_value("cameras", ", ".join(vlm.get("cameras") or [])))
        rows.append(self._render_key_value("target objects", ", ".join(vlm.get("target_objects") or [])))
        if vlm.get("montage"):
            rows.append(
                f"<p><img src='{escape(self._html_path(vlm.get('montage'), dump_dir))}' "
                "style='max-width:100%; border:1px solid #d7dce2'></p>"
            )
        if vlm.get("instruction"):
            rows.append(f"<pre>{escape(str(vlm.get('instruction')))}</pre>")
        rows.append("<div class='grid'>")
        for name, entry in (vlm.get("objects") or {}).items():
            phrases = ", ".join(str(item) for item in entry.get("phrases") or [])
            rows.append("<div class='card'>")
            rows.append(f"<h3>{escape(str(name))}</h3>")
            rows.append(self._render_key_value("best phrase", entry.get("best_phrase")))
            rows.append(self._render_key_value("phrases", phrases))
            rows.append(self._render_key_value("visible", entry.get("visible")))
            rows.append(self._render_key_value("reason", entry.get("reason")))
            rows.append("</div>")
        rows.append("</div></section>")
        return "\n".join(rows)

    def _render_phrase_attempts(self, attempt: Dict, dump_dir: str) -> str:
        rows = [
            "<section class='attempt'>",
            "<h3>VLM Phrase Candidates</h3>",
            self._render_key_value("summary", attempt.get("relative_path")),
            self._render_key_value("selected phrase", attempt.get("selected_phrase")),
            "<table><tr><th>rank</th><th>phrase</th><th>status</th><th>score</th><th>points</th><th>center</th><th>reason</th></tr>",
        ]
        for record in attempt.get("records") or []:
            score_info = record.get("score_info") or {}
            reasons = ", ".join(score_info.get("reasons") or [])
            if record.get("selection_reason"):
                reasons = f"{record.get('selection_reason')}; {reasons}".strip("; ")
            if record.get("error"):
                reasons = record.get("error")
            selected = " *" if record.get("selected") else ""
            rows.append(
                "<tr>"
                f"<td>{escape(str(record.get('rank')))}</td>"
                f"<td>{escape(str(record.get('phrase')))}{selected}</td>"
                f"<td>{escape(str(record.get('status')))}</td>"
                f"<td>{escape(self._format_float(score_info.get('score')))}</td>"
                f"<td>{escape(str(record.get('fused_point_count', '')))}</td>"
                f"<td>{escape(self._format_vec(record.get('fused_object_point_center')))}</td>"
                f"<td>{escape(reasons)}</td>"
                "</tr>"
            )
        rows.extend(["</table>", "</section>"])
        return "\n".join(rows)

    def _render_single_attempt(self, attempt: Dict, dump_dir: str) -> str:
        rows = [
            "<section class='attempt'>",
            f"<h3>Single Object: {escape(str(attempt.get('text')))}</h3>",
            self._render_key_value("summary", attempt.get("relative_path")),
            self._render_key_value("fused points", attempt.get("fused_point_count")),
            self._render_key_value("center", self._format_vec(attempt.get("fused_object_point_center"))),
            self._render_key_value("filters", self._single_filter_text(attempt)),
            "<div class='grid'>",
        ]
        for result in attempt.get("camera_results") or []:
            used = bool(result.get("used_for_fusion"))
            card_class = "used" if used else "rejected"
            status = "used" if used else f"rejected: {result.get('reject_reason')}"
            rows.append(f"<div class='card {card_class}'>")
            rows.append(f"<h3>{escape(str(result.get('camera')))}</h3>")
            if result.get("overlay"):
                rows.append(f"<img src='{escape(self._html_path(result.get('overlay'), dump_dir))}'>")
            rows.append(self._render_key_value("status", status))
            rows.append(self._render_key_value("score", self._format_float(result.get("box_score"))))
            rows.append(self._render_key_value("points", result.get("object_point_count")))
            rows.append(self._render_key_value("mask pixels", result.get("mask_pixels")))
            rows.append(self._render_key_value("center", self._format_vec(result.get("object_point_center"))))
            rows.append(self._render_key_value("label", result.get("label")))
            rows.append(self._render_key_value("box index", result.get("selected_box_index")))
            rows.append(self._render_key_value("candidates", result.get("candidate_count")))
            fallback = result.get("candidate_fallback")
            if fallback:
                rows.append(self._render_key_value("fallback", self._fallback_text(fallback)))
                rows.append(self._render_fallback_table(fallback, dump_dir))
            rows.append("</div>")
        rows.extend(["</div>", "</section>"])
        return "\n".join(rows)

    def _fallback_text(self, fallback: Dict) -> str:
        selected = fallback.get("selected")
        selected_box = fallback.get("selected_source_box_index")
        score = self._format_float(fallback.get("selected_score"))
        reasons = ", ".join(fallback.get("selected_reasons") or [])
        return f"selected={selected}, box={selected_box}, score={score}, {reasons}"

    def _render_fallback_table(self, fallback: Dict, dump_dir: str) -> str:
        rows = [
            "<table><tr><th>box</th><th>score</th><th>points</th><th>mask</th><th>center</th><th>reason</th></tr>"
        ]
        for candidate in fallback.get("candidates") or []:
            reasons = ", ".join(candidate.get("reasons") or [])
            rows.append(
                "<tr>"
                f"<td>{escape(str(candidate.get('source_box_index')))}</td>"
                f"<td>{escape(self._format_float(candidate.get('score')))}</td>"
                f"<td>{escape(str(candidate.get('object_point_count')))}</td>"
                f"<td>{escape(str(candidate.get('mask_pixels')))}</td>"
                f"<td>{escape(self._format_vec(candidate.get('object_point_center')))}</td>"
                f"<td>{escape(reasons)}</td>"
                "</tr>"
            )
        rows.append("</table>")
        return "\n".join(rows)

    def _render_instance_attempt(self, attempt: Dict, dump_dir: str) -> str:
        rows = [
            "<section class='attempt'>",
            f"<h3>Instances: {escape(str(attempt.get('text')))}</h3>",
            self._render_key_value("summary", attempt.get("relative_path")),
            self._render_key_value("candidates", attempt.get("candidate_count")),
            self._render_key_value("accepted candidates", attempt.get("accepted_candidate_count")),
            self._render_key_value("instances", attempt.get("instance_count")),
            self._render_key_value("cluster distance", attempt.get("cluster_distance")),
            "<h3>Detected Instances</h3>",
            "<table><tr><th>id</th><th>points</th><th>center</th><th>cameras</th><th>members</th></tr>",
        ]
        for instance in attempt.get("instances") or []:
            rows.append(
                "<tr>"
                f"<td>{escape(str(instance.get('instance_id')))}</td>"
                f"<td>{escape(str(instance.get('fused_point_count')))}</td>"
                f"<td>{escape(self._format_vec(instance.get('fused_object_point_center')))}</td>"
                f"<td>{escape(', '.join(instance.get('cameras') or []))}</td>"
                f"<td>{escape(str(instance.get('candidate_count')))}</td>"
                "</tr>"
            )
        rows.append("</table><h3>Candidates</h3><div class='grid'>")
        for candidate in attempt.get("candidates") or []:
            accepted = bool(candidate.get("accepted"))
            card_class = "used" if accepted else "rejected"
            status = "accepted" if accepted else f"rejected: {candidate.get('reject_reason')}"
            rows.append(f"<div class='card {card_class}'>")
            rows.append(f"<h3>{escape(str(candidate.get('camera')))} box {escape(str(candidate.get('source_box_index')))}</h3>")
            if candidate.get("overlay"):
                rows.append(f"<img src='{escape(self._html_path(candidate.get('overlay'), dump_dir))}'>")
            rows.append(self._render_key_value("status", status))
            rows.append(self._render_key_value("score", self._format_float(candidate.get("box_score"))))
            rows.append(self._render_key_value("box area", self._format_float(candidate.get("box_area_frac"))))
            rows.append(self._render_key_value("points", candidate.get("object_point_count")))
            rows.append(self._render_key_value("center", self._format_vec(candidate.get("object_point_center"))))
            rows.append(self._render_key_value("label", candidate.get("label")))
            rows.append("</div>")
        rows.extend(["</div>", "</section>"])
        return "\n".join(rows)

    def _render_key_value(self, key: str, value) -> str:
        if value is None:
            value = ""
        return f"<div class='kv'><b>{escape(str(key))}</b> {escape(str(value))}</div>"

    def _single_filter_text(self, attempt: Dict) -> str:
        return (
            f"min_points={attempt.get('min_points')}, "
            f"max_points={attempt.get('max_points')}, "
            f"min_score={attempt.get('min_score')}, "
            f"max_center_z={attempt.get('max_center_z')}, "
            f"selection={attempt.get('selection')}"
        )

    def _format_vec(self, value) -> str:
        if value is None:
            return ""
        try:
            return "[" + ", ".join(f"{float(item):.4f}" for item in value) + "]"
        except Exception:
            return str(value)

    def _format_float(self, value) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def _relative_path(self, path: str, root: str) -> str:
        try:
            return os.path.relpath(path, root)
        except ValueError:
            return path

    def _html_path(self, path: Optional[str], root: str) -> str:
        if not path:
            return ""
        if os.path.isabs(path):
            try:
                return os.path.relpath(path, root)
            except ValueError:
                return path
        return path


class InProcessPerceptionBackend(SubprocessPerceptionBackend):
    """Run VLM + GroundingDINO/SAM2 in the same Python process.

    This backend is meant for the combined Python 3.11 test environment. It
    keeps the detector and segmenter loaded across calls, which avoids the
    repeated conda startup and model loading cost of SubprocessPerceptionBackend.
    """

    def __init__(
        self,
        *args,
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        sam2_model: str = "facebook/sam2.1-hiera-tiny",
        device: Optional[str] = None,
        text_threshold: float = 0.25,
        candidate_fallback_enabled: bool = True,
        candidate_fallback_max_candidates: int = 3,
        candidate_fallback_trigger_points: int = 5000,
        candidate_fallback_trigger_mask_pixels: int = 5000,
        candidate_fallback_trigger_box_area_frac: float = 0.20,
        candidate_fallback_min_score: float = 0.2,
        candidate_fallback_workspace_z_margin: float = 0.65,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grounding_model_name = grounding_model
        self.sam2_model_name = sam2_model
        self.device = device
        self.text_threshold = text_threshold
        self.candidate_fallback_enabled = candidate_fallback_enabled
        self.candidate_fallback_max_candidates = candidate_fallback_max_candidates
        self.candidate_fallback_trigger_points = candidate_fallback_trigger_points
        self.candidate_fallback_trigger_mask_pixels = candidate_fallback_trigger_mask_pixels
        self.candidate_fallback_trigger_box_area_frac = candidate_fallback_trigger_box_area_frac
        self.candidate_fallback_min_score = candidate_fallback_min_score
        self.candidate_fallback_workspace_z_margin = candidate_fallback_workspace_z_margin
        self._grounding_processor = None
        self._grounding_model = None
        self._sam2_predictor = None

    def _get_openai_vlm_descriptions(self, env, manifest_path: str, dump_dir: str) -> Dict:
        instruction = self.vlm_instruction or ""
        if not instruction:
            task_name = env.task.get_name() if env.task is not None else "unknown task"
            instruction = f"Describe task-relevant objects for the RLBench task {task_name}."
        object_names = env.get_object_names()
        cache_key = (
            id(env.latest_obs),
            instruction,
            tuple(object_names),
            self.vlm_object_context,
            self.vlm_model,
            self.vlm_phrase_count,
            self.vlm_panel_size,
        )
        if cache_key in self._vlm_cache:
            return self._vlm_cache[cache_key]

        from .run_openai_vlm_describer import DEFAULT_OPENAI_BASE_URL, describe_scene

        out_dir = os.path.join(dump_dir, "openai_vlm")
        summary = describe_scene(
            manifest_path=manifest_path,
            instruction=instruction,
            objects=object_names,
            out_dir=out_dir,
            model=self.vlm_model,
            cameras=list(self.cameras),
            panel_size=self.vlm_panel_size,
            phrase_count=self.vlm_phrase_count,
            object_context=self.vlm_object_context,
            detail=self.vlm_detail,
            max_output_tokens=900,
            api_key_env=self.vlm_api_key_env,
            base_url=self.vlm_base_url or DEFAULT_OPENAI_BASE_URL,
            api_mode=self.vlm_api_mode,
            timeout=self.timeout,
            reasoning_effort=self.vlm_reasoning_effort,
        )
        self._vlm_cache[cache_key] = summary
        return summary

    def _run_single_query_with_summary(
        self,
        config: Dict,
        manifest_path: str,
        out_dir: str,
    ) -> tuple:
        (
            processor,
            grounding_model,
            sam2_predictor,
            device,
        ) = self._load_models()
        multicam = self._multicam_module()

        os.makedirs(out_dir, exist_ok=True)
        text = config["text"]
        text_for_detector = text if text.strip().endswith(".") else f"{text.strip()}."
        manifest = multicam._load_manifest(manifest_path)
        cameras = multicam._camera_entries(manifest, list(self.cameras))
        if not cameras:
            raise ValueError("No cameras selected.")

        results = []
        fused_points = []
        for camera in cameras:
            result = multicam._run_one_camera(
                camera=camera,
                text=text_for_detector,
                processor=processor,
                grounding_model=grounding_model,
                sam2_predictor=sam2_predictor,
                device=device,
                out_dir=out_dir,
                grounding_scale=self.grounding_scale,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                box_index=config.get("box_index"),
                selection=config.get("selection", "smallest_labeled_area"),
            )
            passes_filters = multicam._passes_filters(
                result,
                int(config.get("min_points", 20)),
                self.min_score,
                config.get("max_center_z"),
                config.get("max_points"),
            )
            if self._should_run_candidate_fallback(result, passes_filters, config):
                fallback = self._run_camera_candidate_fallback(
                    camera=camera,
                    text=text_for_detector,
                    processor=processor,
                    grounding_model=grounding_model,
                    sam2_predictor=sam2_predictor,
                    device=device,
                    out_dir=out_dir,
                    config=config,
                    manifest=manifest,
                    original_result=result,
                )
                if fallback is not None:
                    result = fallback
                    passes_filters = multicam._passes_filters(
                        result,
                        int(config.get("min_points", 20)),
                        self.min_score,
                        config.get("max_center_z"),
                        config.get("max_points"),
                    )

            if passes_filters:
                fused_points.append(np.load(result["object_points"]))
            results.append(result)

        if fused_points:
            fused_points_arr = np.concatenate(fused_points, axis=0)
        else:
            fused_points_arr = np.empty((0, 3), dtype=np.float32)

        fused_points_path = os.path.join(out_dir, "fused_object_points.npy")
        np.save(fused_points_path, fused_points_arr)
        summary = {
            "text": text,
            "manifest": manifest_path,
            "grounding_model": self.grounding_model_name,
            "sam2_model": self.sam2_model_name,
            "grounding_scale": self.grounding_scale,
            "box_threshold": self.box_threshold,
            "selection": config.get("selection", "smallest_labeled_area"),
            "min_points": int(config.get("min_points", 20)),
            "max_points": config.get("max_points"),
            "min_score": self.min_score,
            "max_center_z": config.get("max_center_z"),
            "camera_results": results,
            "fused_object_points": fused_points_path,
            "fused_point_count": int(len(fused_points_arr)),
            "fused_object_point_center": fused_points_arr.mean(axis=0).tolist()
            if len(fused_points_arr)
            else None,
        }
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(
            "[perception] in-process "
            f"text={text!r} fused_points={summary['fused_point_count']} out={out_dir}"
        )
        return fused_points_arr, summary

    def _should_run_candidate_fallback(self, result: Dict, passes_filters: bool, config: Dict) -> bool:
        if not self.candidate_fallback_enabled:
            return False
        if not result.get("accepted"):
            return result.get("reason") not in {"no_boxes"}
        if result.get("reject_reason") in {"too_many_points", "center_z_too_high", "box_too_large"}:
            return True
        if result.get("object_point_count", 0) > self.candidate_fallback_trigger_points:
            return True
        if result.get("mask_pixels", 0) > self.candidate_fallback_trigger_mask_pixels:
            return True
        box_area_frac = self._box_area_frac(result.get("box"), result.get("rgb"))
        if box_area_frac is not None and box_area_frac > self.candidate_fallback_trigger_box_area_frac:
            return True
        if result.get("box") is not None and result.get("rgb") and self._box_touches_image_edge(
            result["box"],
            result["rgb"],
        ):
            return True
        return not passes_filters

    def _run_camera_candidate_fallback(
        self,
        camera: Dict,
        text: str,
        processor,
        grounding_model,
        sam2_predictor,
        device: str,
        out_dir: str,
        config: Dict,
        manifest: Dict,
        original_result: Dict,
    ) -> Optional[Dict]:
        instance_module = self._instance_module()
        candidates = instance_module._run_camera_candidates(
            camera=camera,
            text=text,
            processor=processor,
            grounding_model=grounding_model,
            sam2_predictor=sam2_predictor,
            device=device,
            out_dir=out_dir,
            grounding_scale=self.grounding_scale,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            labeled_only=False,
            max_candidates_per_camera=self.candidate_fallback_max_candidates,
            min_points=int(config.get("min_points", 20)),
            max_points=config.get("max_points"),
            min_score=self.min_score,
            max_center_z=config.get("max_center_z"),
            max_box_area_frac=None,
        )
        if not candidates:
            original_result["candidate_fallback"] = {
                "triggered": True,
                "selected": False,
                "reason": "no_candidates",
            }
            return original_result

        scored = []
        for candidate in candidates:
            score_info = self._score_camera_candidate(candidate, config, manifest)
            candidate["candidate_fallback_score"] = score_info["score"]
            candidate["candidate_fallback_reasons"] = score_info["reasons"]
            scored.append((score_info["score"], candidate, score_info))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate, best_info = scored[0]

        selected = best_score >= self.candidate_fallback_min_score
        fallback_summary = {
            "triggered": True,
            "selected": selected,
            "selected_source_box_index": best_candidate.get("source_box_index") if selected else None,
            "selected_score": best_score,
            "selected_reasons": best_info["reasons"],
            "original": self._camera_result_brief(original_result),
            "candidates": [
                {
                    "source_box_index": candidate.get("source_box_index"),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("candidate_fallback_score"),
                    "reasons": candidate.get("candidate_fallback_reasons"),
                    "box_score": candidate.get("box_score"),
                    "box_area_frac": candidate.get("box_area_frac"),
                    "object_point_count": candidate.get("object_point_count"),
                    "mask_pixels": candidate.get("mask_pixels"),
                    "object_point_center": candidate.get("object_point_center"),
                    "overlay": candidate.get("overlay"),
                    "label": candidate.get("label"),
                }
                for _, candidate, _ in scored
            ],
        }
        if not selected:
            original_result["candidate_fallback"] = fallback_summary
            original_result["accepted"] = False
            original_result["used_for_fusion"] = False
            original_result["reject_reason"] = "candidate_score_low"
            original_result["reason"] = "candidate_score_low"
            return original_result

        result = self._candidate_to_camera_result(best_candidate, candidates)
        result["candidate_fallback"] = fallback_summary
        print(
            "[perception] camera candidate fallback "
            f"camera={camera.get('name')} text={text!r} "
            f"box={best_candidate.get('source_box_index')} score={best_score:.3f} "
            f"reasons={best_info['reasons']}"
        )
        return result

    def _score_camera_candidate(self, candidate: Dict, config: Dict, manifest: Dict) -> Dict:
        return score_camera_candidate_v2(
            candidate,
            config,
            manifest,
            trigger_points=self.candidate_fallback_trigger_points,
            trigger_mask_pixels=self.candidate_fallback_trigger_mask_pixels,
            trigger_box_area_frac=self.candidate_fallback_trigger_box_area_frac,
        )

    def _candidate_to_camera_result(self, candidate: Dict, candidates: List[Dict]) -> Dict:
        return {
            "camera": candidate.get("camera"),
            "accepted": True,
            "rgb": candidate.get("rgb"),
            "point_cloud": candidate.get("point_cloud"),
            "selected_box_index": candidate.get("source_box_index"),
            "box": candidate.get("box"),
            "box_score": candidate.get("box_score"),
            "label": candidate.get("label"),
            "mask_pixels": candidate.get("mask_pixels"),
            "object_point_count": candidate.get("object_point_count"),
            "object_point_center": candidate.get("object_point_center"),
            "mask": candidate.get("mask"),
            "object_points": candidate.get("object_points"),
            "overlay": candidate.get("overlay"),
            "candidates": [
                {
                    "index": item.get("source_box_index"),
                    "box": item.get("box"),
                    "score": item.get("box_score"),
                    "label": item.get("label"),
                    "candidate_fallback_score": item.get("candidate_fallback_score"),
                    "candidate_fallback_reasons": item.get("candidate_fallback_reasons"),
                    "object_point_count": item.get("object_point_count"),
                    "mask_pixels": item.get("mask_pixels"),
                    "object_point_center": item.get("object_point_center"),
                    "overlay": item.get("overlay"),
                }
                for item in candidates
            ],
        }

    def _camera_result_brief(self, result: Dict) -> Dict:
        return {
            "selected_box_index": result.get("selected_box_index"),
            "box_score": result.get("box_score"),
            "object_point_count": result.get("object_point_count"),
            "mask_pixels": result.get("mask_pixels"),
            "object_point_center": result.get("object_point_center"),
            "box": result.get("box"),
            "label": result.get("label"),
            "reject_reason": result.get("reject_reason") or result.get("reason"),
        }

    def _candidate_workspace_max_z(self, manifest: Dict) -> Optional[float]:
        bounds_min = manifest.get("workspace_bounds_min")
        if not bounds_min:
            return None
        return float(bounds_min[2]) + float(self.candidate_fallback_workspace_z_margin)

    def _box_area_frac(self, box, rgb_path: Optional[str]) -> Optional[float]:
        if box is None or not rgb_path:
            return None
        try:
            from PIL import Image

            with Image.open(rgb_path) as image:
                width, height = image.size
            area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
            return area / max(1.0, float(width * height))
        except Exception:
            return None

    def _box_touches_image_edge(self, box, rgb_path: str, margin: float = 1.0) -> bool:
        try:
            from PIL import Image

            with Image.open(rgb_path) as image:
                width, height = image.size
            return self._box_touches_shape_edge(box, [height, width], margin=margin)
        except Exception:
            return False

    def _box_touches_shape_edge(self, box, image_shape, margin: float = 1.0) -> bool:
        if box is None or image_shape is None:
            return False
        height, width = float(image_shape[0]), float(image_shape[1])
        return (
            float(box[0]) <= margin
            or float(box[1]) <= margin
            or float(box[2]) >= width - margin
            or float(box[3]) >= height - margin
        )

    def _run_instance_query(
        self,
        base_name: str,
        instance_index: int,
        manifest_path: str,
        out_dir: str,
        config_override: Optional[Dict] = None,
    ) -> np.ndarray:
        config = config_override or self.INSTANCE_OBJECT_CONFIGS[base_name]
        (
            processor,
            grounding_model,
            sam2_predictor,
            device,
        ) = self._load_models()
        instance_module = self._instance_module()

        os.makedirs(out_dir, exist_ok=True)
        text = config["text"]
        text_for_detector = text if text.strip().endswith(".") else f"{text.strip()}."
        manifest = instance_module._load_manifest(manifest_path)
        cameras = instance_module._camera_entries(manifest, list(self.cameras))
        if not cameras:
            raise ValueError("No cameras selected.")

        candidates = []
        for camera in cameras:
            candidates.extend(
                instance_module._run_camera_candidates(
                    camera=camera,
                    text=text_for_detector,
                    processor=processor,
                    grounding_model=grounding_model,
                    sam2_predictor=sam2_predictor,
                    device=device,
                    out_dir=out_dir,
                    grounding_scale=self.grounding_scale,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    labeled_only=not config.get("include_unlabeled", False),
                    max_candidates_per_camera=config.get("max_candidates_per_camera", 5),
                    min_points=int(config.get("min_points", 10)),
                    max_points=config.get("max_points"),
                    min_score=self.min_score,
                    max_center_z=config.get("max_center_z"),
                    max_box_area_frac=config.get("max_box_area_frac", 0.20),
                )
            )

        clusters = instance_module._cluster_candidates(
            candidates,
            float(config.get("cluster_distance", 0.08)),
        )
        instances = instance_module._save_instances(candidates, clusters, out_dir)
        summary = {
            "text": text,
            "manifest": manifest_path,
            "grounding_model": self.grounding_model_name,
            "sam2_model": self.sam2_model_name,
            "grounding_scale": self.grounding_scale,
            "box_threshold": self.box_threshold,
            "max_candidates_per_camera": config.get("max_candidates_per_camera", 5),
            "include_unlabeled": bool(config.get("include_unlabeled", False)),
            "min_points": int(config.get("min_points", 10)),
            "max_points": config.get("max_points"),
            "min_score": self.min_score,
            "max_center_z": config.get("max_center_z"),
            "max_box_area_frac": config.get("max_box_area_frac", 0.20),
            "cluster_distance": float(config.get("cluster_distance", 0.08)),
            "candidate_count": len(candidates),
            "accepted_candidate_count": sum(1 for candidate in candidates if candidate.get("accepted")),
            "candidates": candidates,
            "instance_count": len(instances),
            "instances": instances,
        }
        summary_path = os.path.join(out_dir, "instances.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        usable_instances = sorted(
            [
                instance
                for instance in instances
                if len(instance.get("cameras", [])) >= config.get("min_instance_cameras", 1)
            ],
            key=lambda inst: inst["fused_object_point_center"][1],
        )
        print(
            "[perception] in-process instances "
            f"text={text!r} candidates={len(candidates)} instances={len(usable_instances)} out={out_dir}"
        )
        if instance_index >= len(usable_instances):
            raise ValueError(
                f"Requested {base_name}{instance_index + 1}, but only found {len(usable_instances)} instances"
            )
        return np.load(usable_instances[instance_index]["fused_object_points"])

    def _load_models(self):
        if (
            self._grounding_processor is not None
            and self._grounding_model is not None
            and self._sam2_predictor is not None
        ):
            return (
                self._grounding_processor,
                self._grounding_model,
                self._sam2_predictor,
                self._device(),
            )

        print(
            "[perception] loading in-process models: "
            f"{self.grounding_model_name}, {self.sam2_model_name}"
        )
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = self._device()
        self._grounding_processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_name
        )
        self._grounding_model = self._grounding_model.to(device).eval()
        self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
            self.sam2_model_name,
            device=device,
        )
        return (
            self._grounding_processor,
            self._grounding_model,
            self._sam2_predictor,
            device,
        )

    def _device(self) -> str:
        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _multicam_module(self):
        from . import run_multicam_grounded_sam2

        return run_multicam_grounded_sam2

    def _instance_module(self):
        from . import run_instance_grounded_sam2

        return run_instance_grounded_sam2
