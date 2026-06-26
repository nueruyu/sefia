from typing import Any, Callable, Type

from pydantic import TypeAdapter, ValidationError

from .._interfaces.model_inspector import ModelInspector
from ._function_models import (
    cache_key,
    create_params_model,
    get_callable_doc,
    get_callable_qualname,
    sanitize_function_name,
)


class PydanticModelInspector(ModelInspector):
    """
    Pydantic-backed implementation for schema generation and validation.
    Supports dataclasses, Pydantic models, primitives, and typing constructs.
    """

    def __init__(self):
        self._schema_cache: dict[Any, dict] = {}
        self._adapter_cache: dict[Any, TypeAdapter] = {}

    def get_type_schema(self, model_type: Type[Any] | Any) -> dict:
        if model_type in self._schema_cache:
            return self._schema_cache[model_type]

        schema = self._get_adapter(model_type).json_schema()
        self._schema_cache[model_type] = schema
        return schema

    def get_function_name(self, func: Callable[..., Any]) -> str:
        return sanitize_function_name(get_callable_qualname(func))

    def get_function_schema(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> dict:
        schema_name = name or self.get_function_name(func)
        cache_key_value = ("function_schema", cache_key(func), schema_name)
        if cache_key_value in self._schema_cache:
            return self._schema_cache[cache_key_value]

        param_model = create_params_model(
            func,
            name=schema_name,
            forbid_extra=True,
        )
        schema = TypeAdapter(param_model).json_schema()

        result = {
            "type": "function",
            "function": {
                "name": schema_name,
                "description": get_callable_doc(func),
                "parameters": schema,
            },
        }
        self._schema_cache[cache_key_value] = result
        return result

    def validate(self, model_type: Type[Any] | Any, data: Any) -> Any:
        try:
            return self._get_adapter(model_type).validate_python(data)
        except ValidationError as e:
            type_name = getattr(model_type, "__name__", str(model_type))
            raise ValueError(f"Model validation failed for {type_name}: {e}") from e

    def _get_adapter(self, model_type: Type[Any] | Any) -> TypeAdapter:
        if model_type not in self._adapter_cache:
            self._adapter_cache[model_type] = TypeAdapter(model_type)
        return self._adapter_cache[model_type]
