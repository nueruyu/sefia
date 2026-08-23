from enum import Enum


class LLMDecisionMode(Enum):
    STRUCTURED_OUTPUT = "structured_output"
    JSON = "json"
    NATIVE_TOOLS = "native_tools"


__all__ = ["LLMDecisionMode"]
