from enum import Enum


class LLMDecisionMode(Enum):
    STRUCTURED_OUTPUT = "structured_output"
    JSON = "json"


__all__ = ["LLMDecisionMode"]
