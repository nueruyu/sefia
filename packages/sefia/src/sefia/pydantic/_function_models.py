import functools
import inspect
import re
from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
)

ExtraPolicy = Literal["forbid", "allow", "ignore"]


class _UnhashableCallableKey:
    __slots__ = ("_obj",)

    def __init__(self, obj: Callable[..., Any]):
        self._obj = obj

    def __hash__(self) -> int:
        return id(self._obj)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnhashableCallableKey) and self._obj is other._obj


def cache_key(func: Callable[..., Any]) -> object:
    try:
        hash(func)
    except TypeError:
        return _UnhashableCallableKey(func)
    return func


def sanitize_function_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.replace(".", "_"))


def get_callable_qualname(func: Callable[..., Any]) -> str:
    if isinstance(func, functools.partial):
        return f"partial_{get_callable_qualname(func.func)}"

    qualname = getattr(func, "__qualname__", None)
    if qualname:
        return qualname

    if not inspect.isclass(func) and callable(func):
        return type(func).__qualname__

    name = getattr(func, "__name__", None)
    if name:
        return name

    return type(func).__qualname__


def _get_callable_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    annotation_source = _get_annotation_source(func)
    if annotation_source is None:
        return {}
    return inspect.get_annotations(annotation_source, eval_str=True)


def _get_annotation_source(func: Callable[..., Any]) -> Callable[..., Any] | None:
    if isinstance(func, functools.partial):
        return _get_annotation_source(func.func)

    if inspect.isfunction(func) or inspect.ismethod(func):
        return inspect.unwrap(func)

    if not inspect.isclass(func) and callable(func):
        call = getattr(func, "__call__", None)
        if call is not None:
            return inspect.unwrap(call)

    return func


def get_callable_doc(func: Callable[..., Any]) -> str:
    if isinstance(func, functools.partial):
        return get_callable_doc(func.func)

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


def _create_params_model(
    func: Callable[..., Any],
    *,
    name: str,
    extra: ExtraPolicy,
) -> type[BaseModel]:
    field_definitions = _build_param_fields(func)
    return create_model(
        f"{name}Params",
        __config__=ConfigDict(extra=extra),
        **field_definitions,
    )


class PydanticFunctionModelFactory:
    """Creates and caches Pydantic models derived from callable signatures."""

    def __init__(self):
        self._params_model_cache: dict[object, type[BaseModel]] = {}

    def params_model(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        extra: ExtraPolicy = "forbid",
    ) -> type[BaseModel]:
        key = (cache_key(func), name, extra)
        if key not in self._params_model_cache:
            self._params_model_cache[key] = _create_params_model(
                func,
                name=name,
                extra=extra,
            )
        return self._params_model_cache[key]


def _build_param_fields(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    type_hints = _get_callable_annotations(func)
    qualname = get_callable_qualname(func)

    params: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(
                f"Tool parameter '{param_name}' on '{qualname}' is "
                "positional-only, but tools must be callable with keyword "
                "arguments."
            )

        if param.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
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

    return cast(dict[str, Any], params)
