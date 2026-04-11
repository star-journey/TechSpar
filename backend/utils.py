"""Shared utility functions."""
import json
import re


def parse_json_response(content: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code blocks and raw prefixes."""
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    for i, c in enumerate(content):
        if c in ("[", "{"):
            try:
                return json.loads(content[i:])
            except json.JSONDecodeError:
                pass
            break

    raise json.JSONDecodeError("No valid JSON found", content, 0)
