from typing import Any, Callable

from typing_extensions import final, override

from ..llm.model_backend import ModelBackend
from ..llm.result_schema import ResultSchema
from .._tool_system import ToolDefinition
from ._function_models import (
    PydanticFunctionModelFactory,
    cache_key,
    get_callable_doc,
    get_callable_qualname,
    sanitize_function_name,
)
from ._result_schema import PydanticResultSchemaFactory


@final
class PydanticModelBackend(ModelBackend):
    """
    Pydantic-backed tool-function inspector and structured-value schema factory.
    Supports dataclasses, Pydantic models, primitives, and typing constructs.
    """

    def __init__(
        self,
        function_model_factory: PydanticFunctionModelFactory | None = None,
    ):
        self._function_model_factory = (
            function_model_factory or PydanticFunctionModelFactory()
        )
        self._result_schema_factory = PydanticResultSchemaFactory()
        self._definition_cache: dict[Any, ToolDefinition] = {}

    @override
    def tool_name(self, func: Callable[..., Any]) -> str:
        return sanitize_function_name(get_callable_qualname(func))

    @override
    def definition(
        self,
        func: Callable[..., Any],
        *,
        name: str,
    ) -> ToolDefinition:
        cache_key_value = (cache_key(func), name)
        cached = self._definition_cache.get(cache_key_value)
        if cached is not None:
            return cached

        param_model = self._function_model_factory.params_model(
            func,
            name=name,
            extra="forbid",
        )
        definition = ToolDefinition(
            name=name,
            description=get_callable_doc(func),
            parameters=param_model.model_json_schema(),
        )
        self._definition_cache[cache_key_value] = definition
        return definition

    @override
    def bind(
        self,
        func: Callable[..., Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        # Shape is already enforced by the step-decision validator; this only
        # coerces values to the callable's declared types. ``extra="allow"``
        # passes any additional keys through (e.g. for ``**kwargs`` handlers).
        param_model = self._function_model_factory.params_model(
            func,
            name="ToolArguments",
            extra="allow",
        )
        validated = param_model.model_validate(arguments)
        # A shallow dump: ``model_dump`` would recursively turn the coerced
        # sub-model/dataclass instances back into dicts, undoing the coercion
        # the callable's annotations asked for.
        return {**dict(validated), **(validated.model_extra or {})}

    @override
    def create(self, python_type: Any) -> ResultSchema:
        return self._result_schema_factory.create(python_type)
