from typing import Any, Callable

from ..interfaces import ModelInspector, ToolCollector
from ..models import ToolConflictError, ToolRegistry


class DefaultToolCollector(ToolCollector):
    """
    The default implementation of ToolCollector.
    Scans an instance and its private attributes for @tool methods.
    Delegates schema generation to a ModelInspector.
    """

    def __init__(self, model_inspector: ModelInspector | None = None):
        if model_inspector is None:
            from ..pydantic.model_inspector import PydanticModelInspector

            model_inspector = PydanticModelInspector()
        self._model_inspector = model_inspector

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()
        self._collect_from_object(instance, registry)

        for attr_name in getattr(instance, "__dict__", {}):
            if attr_name.startswith("_"):
                attr_value = getattr(instance, attr_name, None)
                if attr_value is not None and not callable(attr_value):
                    self._collect_from_object(attr_value, registry)

        return registry

    def _collect_from_object(self, obj: object, registry: ToolRegistry) -> None:
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                method = getattr(obj, name)
                if callable(method) and hasattr(method, "__sefia_tool__"):
                    schema = self._build_schema(method)
                    registry.add(method, schema)
            except ToolConflictError:
                raise
            except Exception:
                continue

    def _build_schema(self, func: Callable[..., Any]) -> dict:
        """
        Generates a JSON schema for a function's parameters via ModelInspector.
        """
        return self._model_inspector.get_schema_for_function(func)
