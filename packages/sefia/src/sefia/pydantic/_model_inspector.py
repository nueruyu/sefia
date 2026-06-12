import functools
import inspect
import re
from typing import Any, Callable, Type, cast

from pydantic import TypeAdapter, ValidationError, create_model

from .._interfaces.model_inspector import ModelInspector


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
        cache_key = self._cache_key(func)
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        sig = inspect.signature(func)
        type_hints = self._get_callable_annotations(func)
        name = self._get_callable_name(func)
        qualname = self._get_callable_qualname(func)

        params: dict[str, tuple[Any, Any]] = {}
        for param_name, param in sig.parameters.items():
            if param.kind not in [
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            ] or param_name in ("self", "cls"):
                continue

            if param_name not in type_hints:
                raise ValueError(
                    f"Tool parameter '{param_name}' on '{qualname}' must have "
                    "a type annotation."
                )

            params[param_name] = (
                type_hints[param_name],
                param.default if param.default is not inspect.Parameter.empty else ...,
            )

        field_definitions = cast(dict[str, Any], params)
        param_model = create_model(f"{name}Params", **field_definitions)
        schema = TypeAdapter(param_model).json_schema()

        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", qualname.replace(".", "_"))

        result = {
            "type": "function",
            "function": {
                "name": sanitized_name,
                "description": self._get_callable_doc(func),
                "parameters": schema,
            },
        }
        self._schema_cache[cache_key] = result
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

    @staticmethod
    def _cache_key(func: Callable[..., Any]) -> Any:
        try:
            hash(func)
        except TypeError:
            return id(func)
        return func

    def _get_callable_name(self, func: Callable[..., Any]) -> str:
        qualname = self._get_callable_qualname(func)
        return re.sub(r"[^a-zA-Z0-9_]", "_", qualname.replace(".", "_"))

    def _get_callable_qualname(self, func: Callable[..., Any]) -> str:
        if isinstance(func, functools.partial):
            return f"partial_{self._get_callable_qualname(func.func)}"

        qualname = getattr(func, "__qualname__", None)
        if qualname:
            return qualname

        if not inspect.isclass(func) and callable(func):
            return type(func).__qualname__

        name = getattr(func, "__name__", None)
        if name:
            return name

        return type(func).__qualname__

    def _get_callable_annotations(self, func: Callable[..., Any]) -> dict[str, Any]:
        annotation_source = self._get_annotation_source(func)
        if annotation_source is None:
            return {}
        return inspect.get_annotations(annotation_source, eval_str=True)

    def _get_annotation_source(
        self, func: Callable[..., Any]
    ) -> Callable[..., Any] | None:
        if isinstance(func, functools.partial):
            return self._get_annotation_source(func.func)

        if inspect.isfunction(func) or inspect.ismethod(func):
            return inspect.unwrap(func)

        if not inspect.isclass(func) and callable(func):
            call = getattr(func, "__call__", None)
            if call is not None:
                return inspect.unwrap(call)

        return func

    def _get_callable_doc(self, func: Callable[..., Any]) -> str:
        if isinstance(func, functools.partial):
            return self._get_callable_doc(func.func)

        if (
            not inspect.isclass(func)
            and not (inspect.isfunction(func) or inspect.ismethod(func))
            and callable(func)
        ):
            return (
                inspect.getdoc(getattr(func, "__call__", None))
                or inspect.getdoc(func)
                or ""
            )

        if inspect.getdoc(func):
            return inspect.getdoc(func) or ""

        return ""
