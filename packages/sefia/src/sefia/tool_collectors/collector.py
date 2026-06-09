from typing import Any, Callable

from ..interfaces import ModelInspector, ToolCollector
from ..models import ToolConflictError, ToolRegistry


class DefaultToolCollector(ToolCollector):
    """
    The default implementation of ToolCollector.

    Tools are discovered structurally, without any marker decorator:

    * Every public method of the dependencies the instance holds in its private
      attributes (for example ``self._web``) is exposed as a tool.
    * Every method of the instance itself is exposed as a tool, including its
      private methods, except for ``@infer`` methods. Those are inference entry
      points rather than tools, and exposing the running one would recurse.

    Schema generation is delegated to a ModelInspector.
    """

    def __init__(self, model_inspector: ModelInspector | None = None):
        if model_inspector is None:
            from ..pydantic.model_inspector import PydanticModelInspector

            model_inspector = PydanticModelInspector()
        self._model_inspector = model_inspector

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()

        # The instance's own methods, private included, but not the @infer
        # inference entry points.
        self._collect_methods(
            instance, registry, include_private=True, skip_infer=True
        )

        # The public methods of each dependency held in a private attribute.
        for attr_name in getattr(instance, "__dict__", {}):
            if not attr_name.startswith("_"):
                continue
            member = getattr(instance, attr_name, None)
            if self._is_tool_provider(member):
                self._collect_methods(
                    member, registry, include_private=False, skip_infer=False
                )

        return registry

    @staticmethod
    def _is_tool_provider(member: object) -> bool:
        """A held member exposes tools only if it is a user-defined object."""
        if member is None or callable(member):
            return False
        return type(member).__module__ != "builtins"

    def _collect_methods(
        self,
        obj: object,
        registry: ToolRegistry,
        *,
        include_private: bool,
        skip_infer: bool,
    ) -> None:
        for name in dir(obj):
            if name.startswith("__"):
                continue
            if name.startswith("_") and not include_private:
                continue
            try:
                method = getattr(obj, name)
            except Exception:
                continue
            if not callable(method):
                continue
            if skip_infer and getattr(method, "__sefia_infer__", False):
                continue
            try:
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
