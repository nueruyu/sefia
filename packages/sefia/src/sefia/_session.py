from typing import Self, Type, TypeVar

import glyff

from ._context import SessionContext, context_var
from ._interfaces import Policy
from ._interfaces.session_store import SessionStore
from ._state_store import StateStore
from .llm.client import LLMClient
from .llm.strategy import LLMInferenceStrategy
from .llm.xml_prompt_formatter import XmlPromptFormatter
from .policies import StagnationPolicy
from .pydantic.json_utils import pydantic_json_default
from .pydantic.model_inspector import PydanticModelInspector
from .tool_collectors.collector import DefaultToolCollector
from .tools import ToolCollector

T = TypeVar("T")


class Session:
    def __init__(
        self,
        llm_client: LLMClient,
        glyff_session: glyff.Session,
        session_store: SessionStore,
        policies: list[Policy] | None = None,
        tool_collector: ToolCollector | None = None,
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self.session_store = session_store
        self._context_token = None
        extra_policies = list(policies) if policies is not None else []
        self.policies: list[Policy] = [
            StagnationPolicy(),
            *extra_policies,
        ]

        model_inspector = PydanticModelInspector()

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
        if self._context is None:
            raise RuntimeError(
                "Cannot get a state store before the session is entered."
            )
        return self._context.get_state_store(key, state_type)

    async def __aenter__(self) -> Self:
        self._context = SessionContext(
            glyff_session=self._glyff_session,
            session_store=self.session_store,
            llm_client=self.llm_client,
            inference_strategy=self._inference_strategy,
            policies=self.policies,
            tool_collector=self._tool_collector,
        )
        self._context_token = context_var.set(self._context)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self._context_token is not None:
            context_var.reset(self._context_token)
            self._context_token = None
