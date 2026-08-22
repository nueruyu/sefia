from .._tool_system import ToolFunctionInspector
from .result_format import ResultFormatFactory


class ModelBackend(ToolFunctionInspector, ResultFormatFactory):
    pass


__all__ = ["ModelBackend"]
