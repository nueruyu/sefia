import inspect
import re
from typing import Any, Callable

from pydantic import TypeAdapter, create_model

from ..interfaces import ToolCollector
from ..models import ToolConflictError, ToolRegistry


class DefaultToolCollector(ToolCollector):
    """
    The default implementation of ToolCollector.
    Scans an instance and its private attributes for @tool methods.
    Caches the resulting ToolRegistry per instance.
    """

    def __init__(self):
        self._schema_cache: dict[Callable, dict] = {}

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

    def _build_schema(self, func: Callable) -> dict:
        """
        Generates a JSON schema for a function's parameters,
        suitable for use with LLM tool-calling features.
        Caches the result.
        """
        if func in self._schema_cache:
            return self._schema_cache[func]

        unwrapped = inspect.unwrap(func)
        sig = inspect.signature(unwrapped)
        type_hints = inspect.get_annotations(unwrapped, eval_str=True)

        params: dict = {
            name: (
                type_hints.get(name, Any),
                param.default if param.default is not inspect.Parameter.empty else ...,
            )
            for name, param in sig.parameters.items()
            if param.kind
            in [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            ]
            and name not in ("self", "cls")
        }

        param_model = create_model(f"{func.__name__}Params", **params)
        schema = TypeAdapter(param_model).json_schema()

        sanitized_name = re.sub(
            r"[^a-zA-Z0-9_-]", "_", func.__qualname__.replace(".", "_")
        )

        result = {
            "type": "function",
            "function": {
                "name": sanitized_name,
                "description": inspect.getdoc(func) or "",
                "parameters": schema,
            },
        }
        self._schema_cache[func] = result
        return result
