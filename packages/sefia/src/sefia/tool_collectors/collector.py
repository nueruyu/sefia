from typing import Any, Callable

from ..interfaces import ModelInspector, ToolCollector
from ..models import ToolConflictError, ToolRegistry
from ..toolify import Toolset


class DefaultToolCollector(ToolCollector):
    """
    The default implementation of ToolCollector.

    Tools are opt-in. A method becomes a tool only when it is explicitly marked
    with ``@tool``, or when an object/function is passed through ``toolify``:

    * The instance's own ``@tool``-marked methods (including private ones) are
      exposed. The running ``@infer`` method is unmarked, so it is never a tool.
    * Each dependency the instance holds in an attribute (public or private)
      contributes either its ``@tool``-marked methods, or — when it is a
      ``Toolset`` from ``toolify`` — every callable it bundles.

    Schema generation is delegated to a ModelInspector.
    """

    def __init__(self, model_inspector: ModelInspector | None = None):
        if model_inspector is None:
            from ..pydantic.model_inspector import PydanticModelInspector

            model_inspector = PydanticModelInspector()
        self._model_inspector = model_inspector

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()

        # The instance's own @tool-marked methods, private included.
        self._collect_marked_methods(instance, registry)

        # Each dependency the instance holds, whether the attribute is public or
        # private.
        for attr_name in getattr(instance, "__dict__", {}):
            member = getattr(instance, attr_name, None)
            if isinstance(member, Toolset):
                for func in member.tools:
                    self._add(func, registry)
            elif self._is_tool_provider(member):
                self._collect_marked_methods(member, registry)

        return registry

    @staticmethod
    def _is_tool_provider(member: object) -> bool:
        """A held member can contribute tools only if it is a user-defined object."""
        if member is None or callable(member):
            return False
        return type(member).__module__ != "builtins"

    def _collect_marked_methods(self, obj: object, registry: ToolRegistry) -> None:
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                method = getattr(obj, name)
            except Exception:
                continue
            if not callable(method):
                continue
            if not getattr(method, "__sefia_tool__", False):
                continue
            self._add(method, registry)

    def _add(self, func: Callable[..., Any], registry: ToolRegistry) -> None:
        try:
            schema = self._build_schema(func)
            registry.add(func, schema)
        except ToolConflictError:
            raise
        except Exception:
            return

    def _build_schema(self, func: Callable[..., Any]) -> dict:
        """
        Generates a JSON schema for a function's parameters via ModelInspector.
        """
        return self._model_inspector.get_schema_for_function(func)
