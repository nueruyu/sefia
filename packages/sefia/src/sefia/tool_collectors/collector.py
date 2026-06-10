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
        # private. Slotted classes have no __dict__, so also gather names from
        # __slots__ across the class hierarchy.
        attr_names = set(getattr(instance, "__dict__", {}))
        for cls in type(instance).__mro__:
            slots = cls.__dict__.get("__slots__", None)
            if not slots:
                continue
            if isinstance(slots, str):
                attr_names.add(slots)
            else:
                attr_names.update(slots)

        for attr_name in attr_names:
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
        # Find @tool-marked method names from the class hierarchy first, then
        # getattr only those. Scanning every name from dir() and calling getattr
        # on each could trigger lazy properties or other side effects on
        # third-party objects. Traversing __wrapped__/__func__ also makes the
        # marker robust to decorator ordering (e.g. @tool under @infer).
        marked_names: set[str] = set()
        for cls in type(obj).__mro__:
            for name, value in cls.__dict__.items():
                current = value
                visited: set[int] = set()
                while current is not None and id(current) not in visited:
                    visited.add(id(current))
                    if getattr(current, "__sefia_tool__", False) is True:
                        marked_names.add(name)
                        break
                    if hasattr(current, "__wrapped__"):
                        current = current.__wrapped__
                    elif hasattr(current, "__func__"):
                        current = current.__func__
                    else:
                        current = None

        for name in marked_names:
            if name.startswith("__"):
                continue
            try:
                method = getattr(obj, name)
            except Exception:
                continue
            if callable(method):
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
