import json
import re
from .llm_output import LLMOutput

_FENCED_JSON = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<content>.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def parse_json_response(text: str, *, allow_surrounding_text: bool) -> LLMOutput:
    raw = text.strip()
    try:
        return LLMOutput.parse_json(raw)
    except json.JSONDecodeError as strict_error:
        if raw.startswith("```"):
            lines = raw.splitlines()
            try:
                return LLMOutput.parse_json("\n".join(lines[1:-1]).strip())
            except json.JSONDecodeError:
                if not allow_surrounding_text:
                    raise
        if not allow_surrounding_text:
            raise

        for match in _FENCED_JSON.finditer(raw):
            try:
                return LLMOutput.parse_json(match.group("content").strip())
            except json.JSONDecodeError:
                continue

        raise strict_error


__all__ = ["parse_json_response"]
