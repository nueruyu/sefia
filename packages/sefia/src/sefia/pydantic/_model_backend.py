from typing import Any, Callable

from .._interfaces.decision_model import (
    DecisionModel,
    DecisionModelBuilder,
    DecisionModelSpec,
)
from .._tool_system import ToolDefinition, ToolFunctionInspector
from ._decision_model import PydanticDecisionModelFactory
from ._function_models import (
    PydanticFunctionModelFactory,
    cache_key,
    get_callable_doc,
    get_callable_qualname,
    sanitize_function_name,
)


class PydanticModelBackend(ToolFunctionInspector, DecisionModelBuilder):
    """
    Pydantic-backed tool-function inspector and decision-model builder.
    Supports dataclasses, Pydantic models, primitives, and typing constructs.
    """

    def __init__(
        self,
        function_model_factory: PydanticFunctionModelFactory | None = None,
    ):
        self._function_model_factory = (
            function_model_factory or PydanticFunctionModelFactory()
        )
        self._decision_model_factory = PydanticDecisionModelFactory()
        self._definition_cache: dict[Any, ToolDefinition] = {}

    def tool_name(self, func: Callable[..., Any]) -> str:
        return sanitize_function_name(get_callable_qualname(func))

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

    def bind(
        self,
        func: Callable[..., Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        # Shape is already enforced upstream by the decision model; this only
        # coerces values to the callable's declared types. ``extra="allow"``
        # passes any additional keys through (e.g. for ``**kwargs`` handlers).
        param_model = self._function_model_factory.params_model(
            func,
            name="ToolArguments",
            extra="allow",
        )
        return param_model.model_validate(arguments).model_dump(mode="python")

    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        return self._decision_model_factory.build(spec)
