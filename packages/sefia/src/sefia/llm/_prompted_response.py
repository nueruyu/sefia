import re
from enum import Enum, auto

_JSON_FENCE = re.compile(
    r"^```(?:json)?[ \t]*\r?\n(?P<content>.*?)^```[ \t]*(?:\r?\n|$)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_JSON_FENCE_START = re.compile(
    r"^```(?:json)?[ \t]*\r?\n",
    re.IGNORECASE | re.MULTILINE,
)
_JSON_FENCE_END = re.compile(r"^```[ \t]*\r?\n")
_JSON_FENCE_END_PREFIX = re.compile(r"`{1,3}[ \t]*\r?")


def extract_prompted_json(text: str) -> str:
    """Extract raw or Markdown-fenced JSON from a prompted response."""
    raw = text.strip()
    if raw.startswith("{"):
        return raw
    fence = _JSON_FENCE.search(raw)
    if fence is None:
        raise ValueError("Prompted response contains no JSON object or JSON fence.")
    return fence.group("content").strip()


class _StreamState(Enum):
    DETECTING = auto()
    FINDING_FENCE = auto()
    RAW = auto()
    FENCED = auto()
    DONE = auto()


class PromptedJsonStreamExtractor:
    """Extract JSON text fragments from a prompted response stream."""

    def __init__(self) -> None:
        self._buffer = ""
        self._state = _StreamState.DETECTING
        self._at_line_start = True

    def feed(self, text: str) -> str:
        if self._state is _StreamState.DONE:
            return ""
        if self._state is _StreamState.RAW:
            return text

        self._buffer += text
        if self._state is _StreamState.DETECTING:
            candidate = self._buffer.lstrip()
            if not candidate:
                return ""
            if candidate.startswith("{"):
                self._state = _StreamState.RAW
                self._buffer = ""
                return candidate
            self._state = _StreamState.FINDING_FENCE

        if self._state is _StreamState.FINDING_FENCE:
            fence = _JSON_FENCE_START.search(self._buffer)
            if fence is None:
                return ""
            self._state = _StreamState.FENCED
            self._buffer = self._buffer[fence.end() :]

        return self._extract_fenced()

    def _extract_fenced(self) -> str:
        extracted: list[str] = []
        while self._buffer:
            if self._at_line_start:
                if _JSON_FENCE_END.match(self._buffer):
                    self._state = _StreamState.DONE
                    self._buffer = ""
                    break
                if _JSON_FENCE_END_PREFIX.fullmatch(self._buffer):
                    break
                self._at_line_start = False

            newline = self._buffer.find("\n")
            if newline < 0:
                extracted.append(self._buffer)
                self._buffer = ""
                break
            extracted.append(self._buffer[: newline + 1])
            self._buffer = self._buffer[newline + 1 :]
            self._at_line_start = True
        return "".join(extracted)


__all__ = ["PromptedJsonStreamExtractor", "extract_prompted_json"]
