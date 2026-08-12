import uuid


class ToolCallIdRegistry:
    """Allocates one stable call id per tool-call index in an LLM step."""

    def __init__(self) -> None:
        self._ids: dict[int, str] = {}

    def get_or_create(self, index: int) -> str:
        return self._ids.setdefault(index, f"call_{uuid.uuid4().hex[:12]}")
