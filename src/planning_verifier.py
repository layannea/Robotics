import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class VerificationResult:
    code: str
    changed: bool = False
    diagnostics: List[str] = field(default_factory=list)


def verify_and_repair_generated_code(
    *,
    lmp_name: str,
    query: str,
    code: str,
    objects: Sequence[str],
) -> VerificationResult:
    """Apply lightweight role-action repairs before executing generated LMP code."""
    if lmp_name != "composer":
        return VerificationResult(code=code)

    result = _repair_button_movable_mismatch(query=query, code=code, objects=objects)
    try:
        ast.parse(result.code)
    except SyntaxError as exc:
        result.diagnostics.append(
            f"discarded verifier repair because it produced invalid Python: {exc}"
        )
        return VerificationResult(code=code, diagnostics=result.diagnostics)
    return result


def _repair_button_movable_mismatch(
    *,
    query: str,
    code: str,
    objects: Sequence[str],
) -> VerificationResult:
    button_objects = _button_like_objects(objects)
    if not button_objects:
        return VerificationResult(code=code)

    query_norm = _normalize_text(query)
    if "button" not in query_norm and "switch" not in query_norm:
        return VerificationResult(code=code)

    lines = code.splitlines()
    parse_assignments = _parse_query_assignments(lines)
    execute_vars = _execute_first_arg_vars(lines)
    risky_vars = set()
    diagnostics = []

    for var_name, query_obj in parse_assignments:
        if var_name not in execute_vars:
            continue
        canonical = _canonicalize_to_object(query_obj, objects)
        if canonical in button_objects:
            risky_vars.add(var_name)
            diagnostics.append(
                "button_or_switch role mismatch: "
                f"execute({var_name}, ...) would treat {query_obj!r} as the movable object"
            )

    if not risky_vars:
        return VerificationResult(code=code)

    repaired_lines = []
    changed = False
    for line in lines:
        assignment = _parse_assignment_line(line)
        if assignment is None:
            repaired_lines.append(line)
            continue
        indent, var_name, _query_obj = assignment
        if var_name in risky_vars:
            repaired_lines.append(f"{indent}{var_name} = parse_query_obj('gripper')")
            changed = True
        else:
            repaired_lines.append(line)

    if changed:
        diagnostics.append(
            "repaired button_or_switch interaction to use gripper-centric motion"
        )
    return VerificationResult(
        code="\n".join(repaired_lines),
        changed=changed,
        diagnostics=diagnostics,
    )


def _parse_query_assignments(lines: Sequence[str]) -> List[Tuple[str, str]]:
    assignments = []
    for line in lines:
        parsed = _parse_assignment_line(line)
        if parsed is None:
            continue
        _indent, var_name, query_obj = parsed
        assignments.append((var_name, query_obj))
    return assignments


def _parse_assignment_line(line: str) -> Optional[Tuple[str, str, str]]:
    match = re.match(
        r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*parse_query_obj\((['\"])(.*?)\3\)\s*$",
        line,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(4)


def _execute_first_arg_vars(lines: Sequence[str]) -> set:
    vars_used = set()
    for line in lines:
        match = re.search(r"\bexecute\(\s*([A-Za-z_][A-Za-z0-9_]*)\b", line)
        if match:
            vars_used.add(match.group(1))
    return vars_used


def _button_like_objects(objects: Sequence[str]) -> set:
    return {
        obj
        for obj in objects
        if any(token in _normalize_text(obj).split() for token in ("button", "switch"))
    }


def _canonicalize_to_object(query_obj: str, objects: Sequence[str]) -> Optional[str]:
    query_norm = _normalize_text(query_obj)
    object_norms = [(obj, _normalize_text(obj)) for obj in objects]
    for obj, obj_norm in object_norms:
        if query_norm == obj_norm:
            return obj

    query_tokens = _tokens(query_norm)
    candidates = []
    for obj, obj_norm in object_norms:
        obj_tokens = _tokens(obj_norm)
        if obj_tokens and obj_tokens.issubset(query_tokens):
            candidates.append((obj, len(obj_tokens)))

    if not candidates:
        for obj, obj_norm in object_norms:
            if obj_norm in {"button", "switch"} and (
                "button" in query_tokens or "switch" in query_tokens
            ):
                candidates.append((obj, 1))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
        return None
    return candidates[0][0]


def _normalize_text(text: str) -> str:
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\b(the|a|an|to|of|on|in|at|from)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set:
    tokens = []
    for token in _normalize_text(text).split():
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return set(tokens)
