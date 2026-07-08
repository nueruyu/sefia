from collections.abc import Hashable
from typing import Self

import glyff

from ._context import ProfileBinding, SessionContext, context_var
from ._interfaces import Policy
from ._interfaces.model_backend import ModelBackend
from ._profiles import Profile
from ._tool_system import ToolCollector
from .llm._client import LLMClient
from .llm._strategy import LLMInferenceStrategy
from .llm._xml_prompt_formatter import XmlPromptFormatter
from .pydantic._json_utils import pydantic_json_default
from .pydantic._model_backend import PydanticModelBackend
from .tool_collectors import DefaultToolCollector


class Session:
    """
    Manages the lifecycle of an inference execution.
    Wraps a glyff.Session and sets up the sefia SessionContext.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        glyff_session: glyff.Session,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
        tool_collector: ToolCollector | None = None,
        model_backend: ModelBackend | None = None,
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self._context_token = None
        self._policies: list[Policy] = list(policies) if policies is not None else []

        model_backend = model_backend or PydanticModelBackend()

        self._tool_collector = tool_collector or DefaultToolCollector(
            model_backend=model_backend
        )
        prompt_formatter = XmlPromptFormatter(json_default=pydantic_json_default)

        # A profile only swaps the client; the rest of the strategy is shared.
        def make_strategy(client: LLMClient) -> LLMInferenceStrategy:
            return LLMInferenceStrategy(
                client,
                model_backend=model_backend,
                prompt_formatter=prompt_formatter,
                json_default=pydantic_json_default,
                stream=stream,
            )

        self._inference_strategy = make_strategy(llm_client)

        self._profiles: dict[Hashable, ProfileBinding] = {}
        for profile in profiles or []:
            if profile.key in self._profiles:
                raise ValueError(f"Duplicate profile key: {profile.key!r}.")
            self._profiles[profile.key] = ProfileBinding(
                strategy=make_strategy(profile.client),
                policies=tuple(profile.policies),
            )

        self._context: SessionContext | None = None

    async def __aenter__(self) -> Self:
        self._context = SessionContext(
            glyff_session=self._glyff_session,
            inference_strategy=self._inference_strategy,
            policies=tuple(self._policies),
            tool_collector=self._tool_collector,
            _profiles=self._profiles,
        )
        self._context_token = context_var.set(self._context)
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        if self._context_token is not None:
            context_var.reset(self._context_token)
            self._context_token = None
