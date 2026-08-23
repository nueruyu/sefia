import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from sefia._tool_system import SignatureToolEntry, ToolEntry
from sefia.event_system import Event, EventPublisher
from sefia.inference import (
    FunctionInfo,
    ToolCallsDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMClient, LLMDecisionMode, LLMInferenceStrategy
from sefia.llm.step_decision import StepDecisionSpec
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default


class MockEventPublisher(EventPublisher):
    def __init__(self) -> None:
        super().__init__(handlers=[])
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class MyOutput:
    name: str
    value: int


@dataclass(frozen=True)
class MyIssue:
    description: str


def search(q: str) -> str:
    """Search for something."""
    raise NotImplementedError


def my_tool(param: int) -> str:
    """A tool taking a single integer parameter."""
    raise NotImplementedError


def chat_tool() -> str:
    """A tool taking no arguments."""
    raise NotImplementedError


_BACKEND = PydanticModelBackend()


def _tool(func: Callable[..., Any]) -> ToolEntry:
    name = _BACKEND.tool_name(func)
    return SignatureToolEntry(
        func,
        name=name,
        schema_source=func,
        inspector=_BACKEND,
    )


def _make_strategy(
    llm_client: LLMClient | None = None,
    *,
    stream: bool = False,
    max_repair_attempts: int = 2,
    decision_mode: LLMDecisionMode = LLMDecisionMode.STRUCTURED_OUTPUT,
) -> LLMInferenceStrategy:
    """The strategy under test, with a stub prompt formatter."""
    formatter = Mock()
    formatter.format_arguments.return_value = "<arguments/>"
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        result_format_factory=PydanticModelBackend(),
        prompt_formatter=formatter,
        json_default=pydantic_json_default,
        stream=stream,
        max_repair_attempts=max_repair_attempts,
        decision_mode=decision_mode,
    )


def _function_info(
    return_type: Any = str,
    arguments: dict[str, Any] | None = None,
    type_hints: dict[str, Any] | None = None,
    instructions: str = "instructions",
) -> FunctionInfo:
    return FunctionInfo(
        qualname="test",
        instructions=instructions,
        bound_arguments=arguments or {},
        type_hints=type_hints or {},
        return_type=return_type,
        args=(),
        kwargs={},
    )


class TestLLMInferenceStrategy:
    def test_build_messages_correctly(self):
        strategy = _make_strategy()
        history = [
            ToolCallsDecision(
                calls=[
                    ToolCallRequest(
                        id="1",
                        name="search",
                        arguments={"q": "日本語の検索クエリ"},
                    )
                ]
            ),
            ToolCallResult(tool_call_id="1", result="見つかりました"),
        ]

        spec = StepDecisionSpec.for_inference(
            name="StepDecision", output_type=str, tools=[_tool(search)]
        )
        messages = strategy._build_messages(
            _function_info(arguments={"arg": "val"}),
            history,
            spec,
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "user"
        assert messages[2].tool_calls is None
        decision = json.loads(str(messages[2].content))
        assert decision == {
            "decision": "tool_calls",
            "tool_calls": [
                {
                    "id": "1",
                    "name": "search",
                    "arguments": {"q": "日本語の検索クエリ"},
                }
            ],
        }
        assert "日本語の検索クエリ" in str(messages[2].content)
        assert "\\u65e5" not in str(messages[2].content)
        assert "見つかりました" in str(messages[3].content)
        assert "\\u898b" not in str(messages[3].content)
        assert json.loads(str(messages[3].content)) == {
            "tool_call_result": {
                "tool_call_id": "1",
                "result": "見つかりました",
            }
        }

    def test_build_messages_does_not_assume_the_formatter_syntax(self):
        strategy = _make_strategy()
        formatter = cast(Any, strategy._prompt_formatter)
        formatter.format_arguments.return_value = "formatted arguments"

        messages = strategy._build_messages(
            _function_info(arguments={"arg": "val"}),
            [],
            StepDecisionSpec.for_inference(
                name="StepDecision", output_type=str, tools=[]
            ),
        )

        assert messages[1].content == "formatted arguments"

    def test_json_mode_hides_internal_call_ids_from_history(self):
        strategy = _make_strategy(decision_mode=LLMDecisionMode.JSON)
        history = [
            ToolCallsDecision(
                calls=[
                    ToolCallRequest(
                        id="internal-id",
                        name="search",
                        arguments={"q": "query"},
                    )
                ]
            ),
            ToolCallResult(tool_call_id="internal-id", result="found"),
        ]
        spec = StepDecisionSpec.for_inference(
            name="StepDecision", output_type=str, tools=[_tool(search)]
        )

        messages = strategy._build_messages(_function_info(), history, spec)

        assert json.loads(str(messages[2].content)) == {
            "decision": "tool_calls",
            "tool_calls": [{"name": "search", "arguments": {"q": "query"}}],
        }
        assert json.loads(str(messages[3].content)) == {
            "tool_call_result": {
                "name": "search",
                "arguments": {"q": "query"},
                "result": "found",
            }
        }

    def test_build_messages_adds_only_result_field_instructions(self):
        strategy = _make_strategy()

        spec = StepDecisionSpec.for_inference(
            name="StepDecision", output_type=list[MyIssue], tools=[]
        )
        messages = strategy._build_messages(
            _function_info(return_type=list[MyIssue]),
            [],
            spec,
        )

        assert messages[0].content == (
            "instructions\n\nSet `decision` to `result` and populate `result` "
            "with the non-null task result."
        )
