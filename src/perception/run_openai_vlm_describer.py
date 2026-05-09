import argparse
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageDraw


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_CAMERAS = ("front", "overhead", "wrist")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _camera_entries(manifest: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
    requested = set(names)
    return [entry for entry in manifest["cameras"] if entry["name"] in requested]


def _split_objects(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _make_montage(cameras: List[Dict[str, Any]], out_path: str, panel_size: int) -> str:
    images = []
    for camera in cameras:
        image = Image.open(camera["rgb"]).convert("RGB")
        image = image.resize((panel_size, panel_size), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        label = camera["name"]
        bbox = draw.textbbox((0, 0), label)
        pad = 4
        draw.rectangle([0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad], fill=(0, 0, 0))
        draw.text((pad, pad), label, fill=(255, 255, 255))
        images.append(image)

    montage = Image.new("RGB", (panel_size * len(images), panel_size), (0, 0, 0))
    for idx, image in enumerate(images):
        montage.paste(image, (idx * panel_size, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path)
    return out_path


def _image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_prompt(
    instruction: str,
    objects: List[str],
    cameras: List[str],
    object_context: str,
    phrase_count: int,
) -> str:
    object_lines = "\n".join(f"- {obj}" for obj in objects)
    camera_text = ", ".join(cameras)
    context_text = ""
    if object_context.strip():
        context_text = f"""
Additional object context:
{object_context.strip()}
"""
    return f"""You are the visual scene parser for a robot perception pipeline.

The image is a montage from these RLBench camera views: {camera_text}. Each panel has its camera name in the top-left corner.

Task instruction:
{instruction}

Target object names:
{object_lines}
{context_text}

Your job:
1. Infer each target object's task role from the instruction and context.
2. Inspect the image and describe the visible physical object matching that role.
3. Produce {phrase_count} short detector-friendly phrases for GroundingDINO/SAM2.
4. Choose one canonical role from this list: movable_object, target_receptacle, button_or_switch, handle_or_part, obstacle, surface_or_region, tool, unknown.

Important:
- Target names may be semantic roles, not literal visual categories.
- Use task relationships to disambiguate objects before describing appearance.
- Do not let a receptacle/container and an item placed into it share the same phrases.
- Do not use table, robot arm, gripper, or camera names as grounding phrases.
- Use only visible appearance in phrases: color, shape, material, and object type.
- Keep each phrase under 8 words.
- Prefer concrete nouns such as basket, bin, button, tomato, cube, debris, paper, trash, object, or container.
- Include multiple phrase variants because GroundingDINO may prefer different wording.
- Make the first phrase the best single phrase.
- Add negative phrases for likely distractors that should not be selected for this target.
- Use JSON only, no markdown.

Return exactly this JSON shape:
{{
  "objects": {{
    "object_name": {{
      "role": "brief task role",
      "role_category": "one canonical role",
      "visible": true,
      "best_phrase": "best detector phrase",
      "phrases": [
        "detector phrase 1",
        "detector phrase 2"
      ],
      "negative_phrases": [
        "distractor phrase 1"
      ],
      "reason": "brief visual reason"
    }}
  }}
}}
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _response_output_text(response: Dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text") and "text" in content:
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def _call_openai_responses(
    prompt: str,
    image_path: str,
    model: str,
    base_url: str,
    detail: str,
    max_output_tokens: int,
    api_key: str,
    timeout: int,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image_path),
                        "detail": detail,
                    },
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    responses_url = base_url.rstrip("/") + "/responses"
    response = requests.post(
        responses_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI Responses API failed with HTTP {response.status_code}:\n{response.text}"
        )
    return response.json()


def _call_openai_chat_completions(
    prompt: str,
    image_path: str,
    model: str,
    base_url: str,
    detail: str,
    max_output_tokens: int,
    api_key: str,
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(image_path),
                            "detail": detail,
                        },
                    },
                ],
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    chat_url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(
        chat_url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI Chat Completions API failed with HTTP {response.status_code}:\n{response.text}"
        )
    return response.json()


def _chat_output_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [
            item.get("text", "")
            for item in content
            if item.get("type") in ("text", "output_text")
        ]
        return "\n".join(chunks).strip()
    return ""


def describe_scene(
    manifest_path: str,
    instruction: str,
    objects: List[str],
    out_dir: str,
    model: str,
    cameras: List[str],
    panel_size: int,
    phrase_count: int,
    object_context: str,
    detail: str,
    max_output_tokens: int,
    api_key_env: str,
    base_url: str,
    api_mode: str,
    timeout: int,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is not set.")

    manifest = _load_manifest(manifest_path)
    selected_cameras = _camera_entries(manifest, cameras)
    if not selected_cameras:
        raise ValueError("No cameras selected from manifest.")

    montage_path = _make_montage(
        selected_cameras,
        os.path.join(out_dir, "vlm_montage.png"),
        panel_size=panel_size,
    )
    prompt = _build_prompt(
        instruction=instruction,
        objects=objects,
        cameras=[camera["name"] for camera in selected_cameras],
        object_context=object_context,
        phrase_count=phrase_count,
    )
    if api_mode == "responses":
        raw_response = _call_openai_responses(
            prompt=prompt,
            image_path=montage_path,
            model=model,
            base_url=base_url,
            detail=detail,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
        raw_output = _response_output_text(raw_response)
    elif api_mode == "chat":
        raw_response = _call_openai_chat_completions(
            prompt=prompt,
            image_path=montage_path,
            model=model,
            base_url=base_url,
            detail=detail,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            timeout=timeout,
        )
        raw_output = _chat_output_text(raw_response)
    else:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    parsed = _extract_json(raw_output)

    result = {
        "model": model,
        "manifest": manifest_path,
        "instruction": instruction,
        "objects": objects,
        "object_context": object_context,
        "cameras": [camera["name"] for camera in selected_cameras],
        "montage": montage_path,
        "detail": detail,
        "base_url": base_url,
        "api_mode": api_mode,
        "phrase_count": phrase_count,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_output": parsed,
        "response_id": raw_response.get("id"),
        "usage": raw_response.get("usage"),
    }
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "vlm_descriptions.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["summary"] = summary_path
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Use OpenAI GPT-5.4 vision to create GroundingDINO prompt ensembles from RLBench RGB dumps."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--objects", required=True, help="Comma-separated target object names.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--panel-size", type=int, default=384)
    parser.add_argument("--phrase-count", type=int, default=4)
    parser.add_argument("--object-context", default="")
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        help="OpenAI-compatible base URL, e.g. http://localhost:8317/v1.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("responses", "chat"),
        default="responses",
        help="Use Responses API or Chat Completions API.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="low",
    )
    args = parser.parse_args()

    result = describe_scene(
        manifest_path=args.manifest,
        instruction=args.instruction,
        objects=_split_objects(args.objects),
        out_dir=args.out_dir,
        model=args.model,
        cameras=args.cameras,
        panel_size=args.panel_size,
        phrase_count=args.phrase_count,
        object_context=args.object_context,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        api_mode=args.api_mode,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
