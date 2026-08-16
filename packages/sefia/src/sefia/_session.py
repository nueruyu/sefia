from collections.abc import Hashable
from typing import Self

import glyff
from typing_extensions import final

from ._context import ProfileBinding, SessionContext, context_var
from ._interfaces import Policy
from .llm.structured_output import StructuredValueSchemaFactory
from ._interfaces.history_storage import HistoryStorage
from ._profiles import Profile
from ._tool_system import ToolCollector, ToolFunctionInspector
from .history_storages import GlyffHistoryStorage
from .llm._client import LLMClient
from .llm._strategy import LLMInferenceStrategy
from .llm._xml_prompt_formatter import XmlPromptFormatter
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
        inspector: ToolFunctionInspector | None = None,
        structured_value_schema_factory: StructuredValueSchemaFactory | None = None,
        stream: bool = False,
        history_storage: HistoryStorage | None = None,
        max_repair_attempts: int = 2,
    ):
        self.llm_client = llm_client
        self._glyff_session = glyff_session
        self._context_token = None
        self._policies: list[Policy] = list(policies) if policies is not None else []
        self._history_storage = history_storage or GlyffHistoryStorage()

        # One Pydantic backend supplies callable inspection and result schemas
        # for whichever default seam the caller did not replace.
        if inspector is None or structured_value_schema_factory is None:
            default_backend = PydanticModelBackend()
            inspector = inspector or default_backend
            structured_value_schema_factory = (
                structured_value_schema_factory or default_backend
            )

        self._tool_collector = tool_collector or DefaultToolCollector(
            inspector=inspector
        )
        prompt_formatter = XmlPromptFormatter(json_default=pydantic_json_default)

        # A profile only swaps the client; the rest of the strategy is shared.
        def make_strategy(client: LLMClient) -> LLMInferenceStrategy:
            return LLMInferenceStrategy(
                client,
                structured_value_schema_factory=structured_value_schema_factory,
                prompt_formatter=prompt_formatter,
                json_default=pydantic_json_default,
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
