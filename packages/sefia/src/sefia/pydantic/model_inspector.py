import inspect
import re
from typing import Any, Callable, Type, cast

from pydantic import TypeAdapter, ValidationError, create_model

from ..interfaces.model_inspector import ModelInspector


class PydanticModelInspector(ModelInspector):
    """
    Pydantic-backed implementation for schema generation and validation.
    Supports dataclasses, Pydantic models, primitives, and typing constructs.
    """

    def __init__(self):
        self._schema_cache: dict[Any, dict] = {}
        self._adapter_cache: dict[Any, TypeAdapter] = {}

    def get_schema_for_type(self, model_type: Type[Any] | Any) -> dict:
        if model_type in self._schema_cache:
            return self._schema_cache[model_type]

        schema = self._get_adapter(model_type).json_schema()
        self._schema_cache[model_type] = schema
        return schema

    def get_schema_for_function(self, func: Callable[..., Any]) -> dict:
        if func in self._schema_cache:
            return self._schema_cache[func]

        unwrapped = inspect.unwrap(func)
        sig = inspect.signature(unwrapped)
        type_hints = inspect.get_annotations(unwrapped, eval_str=True)

        params: dict[str, tuple[Any, Any]] = {
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

        field_definitions = cast(dict[str, Any], params)
        param_model = create_model(f"{func.__name__}Params", **field_definitions)
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

    def validate_and_create(self, model_type: Type[Any] | Any, data: Any) -> Any:
        try:
            return self._get_adapter(model_type).validate_python(data)
        except ValidationError as e:
            type_name = getattr(model_type, "__name__", str(model_type))
            raise ValueError(f"Model validation failed for {type_name}: {e}") from e

    def _get_adapter(self, model_type: Type[Any] | Any) -> TypeAdapter:
        if model_type not in self._adapter_cache:
            self._adapter_cache[model_type] = TypeAdapter(model_type)
        return self._adapter_cache[model_type]
