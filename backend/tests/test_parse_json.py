"""Tests for parse_json_response, focused on unescaped-LaTeX backslash repair."""
from backend.utils import parse_json_response


def test_plain_valid_json_unaffected():
    assert parse_json_response('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}
    assert parse_json_response("[1, 2, 3]") == [1, 2, 3]


def test_correctly_escaped_latex_preserved():
    # Model followed the prompt and doubled the backslashes -> already valid JSON.
    raw = '[{"id": 1, "question": "$\\\\log \\\\pi_\\\\theta$"}]'
    result = parse_json_response(raw)
    assert result[0]["question"] == "$\\log \\pi_\\theta$"


def test_lone_backslash_latex_repaired():
    # Model forgot to escape -> invalid JSON; lone backslashes before non-escape
    # letters are repaired so the batch parses instead of crashing.
    raw = '[{"id": 1, "question": "$\\log \\pi \\cdot \\sum_i$"}]'
    result = parse_json_response(raw)
    assert result[0]["question"] == "$\\log \\pi \\cdot \\sum_i$"


def test_lone_backslash_inside_code_fence():
    raw = '```json\n[{"q": "$\\log x$"}]\n```'
    result = parse_json_response(raw)
    assert result[0]["q"] == "$\\log x$"


def test_lone_backslash_with_leading_prose():
    raw = 'Sure, here is the JSON:\n[{"q": "$\\alpha + \\gamma$"}]'
    result = parse_json_response(raw)
    assert result[0]["q"] == "$\\alpha + \\gamma$"


def test_real_json_escapes_survive_repair():
    # Genuine \n newline and \" quote (e.g. markdown reference_answer) must be
    # preserved untouched by the repair pass.
    raw = '{"text": "line1\\nline2 \\"q\\""}'
    result = parse_json_response(raw)
    assert result["text"] == 'line1\nline2 "q"'
