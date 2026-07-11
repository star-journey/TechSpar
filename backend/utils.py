"""Shared utility functions."""
import json
import re
from pathlib import Path


def resolve_path_within(root: Path, *parts: str) -> Path:
    """Resolve a path and reject absolute/traversing components outside root."""
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("path escapes its allowed directory")
    return candidate


def safe_child_path(root: Path, filename: str) -> Path:
    """Resolve one plain filename below root; nested paths are not accepted."""
    if not filename or filename in {".", ".."} or Path(filename).name != filename:
        raise ValueError("filename must not contain a path")
    return resolve_path_within(root, filename)


_JSON_ESCAPE_OR_BACKSLASH = re.compile(r'\\(["\\/bfnrtu])|\\')


def _repair_backslashes(s: str) -> str:
    """Escape lone backslashes so unescaped LaTeX (e.g. ``$\\log \\pi_\\theta$``)
    becomes valid JSON, while leaving already-valid escape sequences untouched.

    LLMs asked to embed LaTeX inside JSON strings frequently forget to double the
    backslash, which makes ``json.loads`` reject the whole payload.
    """
    return _JSON_ESCAPE_OR_BACKSLASH.sub(
        lambda m: m.group(0) if m.group(1) else "\\\\", s
    )


def _loads(s: str):
    """``json.loads`` with a backslash-repair retry for unescaped LaTeX.

    The common slip is a model emitting LaTeX with lone backslashes (``$\\log
    \\pi$``) instead of the escaped ``\\\\log``. We try the raw payload first so
    correctly-escaped JSON is untouched, then repair and retry. Repair fixes lone
    backslashes before non-escape characters (most LaTeX commands); it cannot
    recover commands whose first letter is itself a valid JSON escape
    (``\\theta`` → tab), which is why the prompt still instructs double-escaping.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(_repair_backslashes(s))


def parse_json_response(content: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks and raw prefixes."""
    content = content.strip()

    try:
        return _loads(content)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
    if m:
        try:
            return _loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    for i, c in enumerate(content):
        if c in ("[", "{"):
            try:
                return _loads(content[i:])
            except json.JSONDecodeError:
                pass
            break

    raise json.JSONDecodeError("No valid JSON found", content, 0)
