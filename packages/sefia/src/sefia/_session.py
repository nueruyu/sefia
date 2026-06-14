from typing import Self, Type, TypeVar

import glyff

from ._context import SessionContext, context_var
from ._interfaces import Policy
from ._interfaces.model_inspector import ModelInspector
from ._interfaces.session_store import SessionStore
from ._state_store import StateStore
from ._tool_system import ToolCollector
from .llm._client import LLMClient
from .llm._strategy import LLMInferenceStrategy
from .llm._xml_prompt_formatter import XmlPromptFormatter
from .policies import StagnationPolicy
from .pydantic._model_inspector import PydanticModelInspector
from .pydantic.json_utils import pydantic_json_default
from .tool_collectors import DefaultToolCollector

T = TypeVar("T")


class Session:
    """
    Manages the lifecycle of an inference execution.
    Wraps a glyff.Session and sets up the sefia SessionContext.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        glyff_session: glyff.Session,
        session_store: SessionStore,
        policies: list[Policy] | None = None,
        tool_collector: ToolCollector | None = None,
        model_inspector: ModelInspector | None = None,
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self.session_store = session_store
        self._context_token = None
        self._policies: list[Policy] = [
            StagnationPolicy(),
            *(policies if policies is not None else []),
        ]

        model_inspector = model_inspector or PydanticModelInspector()

        self._tool_collector = tool_collector or DefaultToolCollector(
            model_inspector=model_inspector
        )
        prompt_formatter = XmlPromptFormatter(json_default=pydantic_json_default)
        self._inference_strategy = LLMInferenceStrategy(
            llm_client,
            model_inspector=model_inspector,
            prompt_formatter=prompt_formatter,
            json_default=pydantic_json_default,
            stream=stream,
        )
        self._context: SessionContext | None = None

    def get_state_store(self, key: str, state_type: Type[T]) -> StateStore[T]:
        """
        Gets a StateStore for the given key and type, which can be used to
        manage persistent state within the session.
        """
        if self._context is None:
            raise RuntimeError(
                "Cannot get a state store before the session is entered."
            )
        return self._context.get_state_store(key, state_type)

    async def __aenter__(self) -> Self:
        self._context = SessionContext(
            glyff_session=self._glyff_session,
            session_store=self.session_store,
            inference_strategy=self._inference_strategy,
            policies=tuple(self._policies),
            tool_collector=self._tool_collector,
        )
        self._context_token = context_var.set(self._context)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self._context_token is not None:
            context_var.reset(self._context_token)
            self._context_token = None
