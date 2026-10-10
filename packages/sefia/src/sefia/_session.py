from collections.abc import Hashable, Sequence
from typing import Self

import glyff
from typing_extensions import final

from ._context import ProfileBinding, SessionContext, context_var
from ._interfaces import Policy
from ._interfaces.history_storage import HistoryStorage
from ._profiles import Profile
from ._tool_system import ToolCollector
from .history_storages import GlyffHistoryStorage
from .llm._client import LLMClient
from .llm._json import JsonMaterializer
from .llm._markdown_prompt_renderer import MarkdownPromptRenderer
from .llm._message_composer import MessageComposer
from .llm._prompt_renderer import PromptRenderer
from .llm._strategy import LLMInferenceStrategy
from .llm.result_format import ResultFormatFactory
from .llm.transports import DecisionTransport, StructuredDecisionTransport
from .pydantic import (
    PydanticJsonMaterializer,
    PydanticResultFormatFactory,
)
from .tool_collectors import DefaultToolCollector


@final
class Session:
    """
    Manages the lifecycle of an inference execution.
    Wraps a glyff.Session and sets up the sefia SessionContext.

    Result-format creation and JSON materialization are independent strategy
    dependencies. Custom tool inspection is configured on ``DefaultToolCollector``.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        glyff_session: glyff.Session,
        policies: list[Policy] | None = None,
        profiles: list[Profile] | None = None,
        tool_collector: ToolCollector | None = None,
        result_format_factory: ResultFormatFactory | None = None,
        json_materializer: JsonMaterializer | None = None,
        stream: bool = False,
        history_storage: HistoryStorage | None = None,
        max_repair_attempts: int = 2,
        prompt_renderer: PromptRenderer | None = None,
        decision_transport: DecisionTransport | None = None,
        message_composers: Sequence[MessageComposer] = (),
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self._context_token = None
        self._policies: list[Policy] = list(policies) if policies is not None else []
        self._history_storage = history_storage or GlyffHistoryStorage()

        if tool_collector is None:
            tool_collector = DefaultToolCollector()
        if result_format_factory is None:
            result_format_factory = PydanticResultFormatFactory()
        if json_materializer is None:
            json_materializer = PydanticJsonMaterializer()

        self._tool_collector = tool_collector
        if prompt_renderer is None:
            prompt_renderer = MarkdownPromptRenderer()
        if decision_transport is None:
            decision_transport = StructuredDecisionTransport()
        message_composers = tuple(message_composers)

        # A profile only swaps the client; the rest of the strategy is shared.
        def make_strategy(client: LLMClient) -> LLMInferenceStrategy:
            return LLMInferenceStrategy(
                client,
                result_format_factory=result_format_factory,
                json_materializer=json_materializer,
                prompt_renderer=prompt_renderer,
                decision_transport=decision_transport,
                message_composers=message_composers,
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
