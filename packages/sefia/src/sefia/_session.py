from collections.abc import Hashable
from typing import Self

import glyff
from typing_extensions import final

from ._context import ProfileBinding, SessionContext, context_var
from ._interfaces import Policy
from .llm.model_backend import ModelBackend
from ._interfaces.history_storage import HistoryStorage
from ._profiles import Profile
from ._tool_system import ToolCollector
from .history_storages import GlyffHistoryStorage
from .llm._client import LLMClient
from .llm._strategy import LLMInferenceStrategy
from .llm._markdown_prompt_renderer import MarkdownPromptRenderer
from .llm._prompt_renderer import PromptRenderer
from .llm.transports import DecisionTransport, StructuredDecisionTransport
from .pydantic._json_utils import pydantic_json_default
from .pydantic._model_backend import PydanticModelBackend
from .tool_collectors import DefaultToolCollector


@final
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
        history_storage: HistoryStorage | None = None,
        max_repair_attempts: int = 2,
        prompt_renderer: PromptRenderer | None = None,
        decision_transport: DecisionTransport | None = None,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self._context_token = None
        self._policies: list[Policy] = list(policies) if policies is not None else []
        self._history_storage = history_storage or GlyffHistoryStorage()

        model_backend = model_backend or PydanticModelBackend()

        self._tool_collector = tool_collector or DefaultToolCollector(
            inspector=model_backend
        )
        prompt_renderer = prompt_renderer or MarkdownPromptRenderer(
            json_default=pydantic_json_default
        )
        decision_transport = decision_transport or StructuredDecisionTransport()

        # A profile only swaps the client; the rest of the strategy is shared.
        def make_strategy(client: LLMClient) -> LLMInferenceStrategy:
            return LLMInferenceStrategy(
                client,
                result_format_factory=model_backend,
                prompt_renderer=prompt_renderer,
                decision_transport=decision_transport,
                stream=stream,
                max_repair_attempts=max_repair_attempts,
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
            history_storage=self._history_storage,
            _profiles=self._profiles,
        )
        self._context_token = context_var.set(self._context)
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object | None,
    ) -> None:
        if self._context_token is not None:
            context_var.reset(self._context_token)
            self._context_token = None
