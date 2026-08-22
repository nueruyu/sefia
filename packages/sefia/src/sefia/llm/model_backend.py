from .._tool_system import ToolFunctionInspector
from .result_schema import ResultSchemaFactory


class ModelBackend(ToolFunctionInspector, ResultSchemaFactory):
    pass


__all__ = ["ModelBackend"]
