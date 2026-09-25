import json
import re

from sefia.llm.json import JsonCompatible


def markdown_fence(content: str) -> str:
    longest_run = max(
        (len(match.group()) for match in re.finditer(r"`+", content)),
        default=0,
    )
    return "`" * max(3, longest_run + 1)


def _code_block(content: str, language: str) -> str:
    fence = markdown_fence(content)
    return f"{fence}{language}\n{content}\n{fence}"


def text_block(value: str) -> str:
    return _code_block(value, "text")


def json_block(value: JsonCompatible) -> str:
    return _code_block(json.dumps(value, ensure_ascii=False, indent=2), "json")


__all__ = ["json_block", "markdown_fence", "text_block"]
