import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock

from sefia._tool_system import SignatureToolEntry, ToolEntry
from sefia.event_system import Event, EventPublisher
from sefia.inference import (
    FunctionInfo,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMClient, LLMInferenceStrategy
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
) -> LLMInferenceStrategy:
    """The strategy under test, with a stub prompt formatter."""
    formatter = Mock()
    formatter.format_arguments.return_value = "<arguments/>"
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        decision_builder=PydanticModelBackend(),
        prompt_formatter=formatter,
        json_default=pydantic_json_default,
        stream=stream,
        max_repair_attempts=max_repair_attempts,
    )


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _decision_branch(schema: dict[str, Any], decision: str) -> dict[str, Any]:
    if schema.get("properties", {}).get("decision", {}).get("const") == decision:
        return schema

    for candidate in schema["oneOf"]:
        branch = _resolve(candidate, schema)
        if branch["properties"]["decision"]["const"] == decision:
            return branch

    raise AssertionError(f"Decision branch not found: {decision}")


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
            ToolCallDecision(
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

        director = strategy._create_director(str, [_tool(search)])
        messages = strategy._build_messages(
            _function_info(arguments={"arg": "val"}),
            history,
            director,
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"
        tool_calls = messages[2].tool_calls
        assert tool_calls is not None
        tool_arguments = tool_calls[0]["function"]["arguments"]
        assert "日本語の検索クエリ" in tool_arguments
        assert "\\u65e5" not in tool_arguments
        assert json.loads(tool_arguments) == {"q": "日本語の検索クエリ"}
        assert "見つかりました" in str(messages[3].content)
        assert "\\u898b" not in str(messages[3].content)
        assert json.loads(str(messages[3].content)) == "見つかりました"

    def test_build_decision_schema_hoists_nested_definitions(self):
        strategy = _make_strategy()

        director = strategy._create_director(list[MyIssue], [_tool(search)])
        schema = director.build_decision_schema().schema

        assert "MyIssue" in schema["$defs"]
        result_branch = _decision_branch(schema, "result")
        result_schema = result_branch["properties"]["result"]
        assert "$defs" not in result_schema
        assert result_schema["items"]["$ref"] == "#/$defs/MyIssue"
        # tool_calls items reference a per-tool call model in $defs, not an inline
        # blob, so each tool's arguments stay constrained by its own schema.
        tool_calls_branch = _decision_branch(schema, "tool_calls")
        tool_calls_array = tool_calls_branch["properties"]["tool_calls"]
        tool_call_ref = tool_calls_array["items"]["$ref"].split("/")[-1]
        assert tool_call_ref in schema["$defs"]
        assert tool_call_ref.endswith("ToolCall")

    def test_build_decision_schema_requires_non_null_result_without_tools(self):
        strategy = _make_strategy()

        director = strategy._create_director(list[MyIssue], [])
        schema = director.build_decision_schema().schema
        result_branch = _decision_branch(schema, "result")

        assert schema["required"] == ["decision", "result"]
        assert result_branch["required"] == ["decision", "result"]
        assert result_branch["properties"]["decision"]["const"] == "result"
        result_schema = result_branch["properties"]["result"]
        assert result_schema["type"] == "array"
        assert "oneOf" not in result_schema

    def test_build_decision_schema_uses_decision_discriminator_with_tools(self):
        strategy = _make_strategy()

        director = strategy._create_director(list[MyIssue], [_tool(search)])
        schema = director.build_decision_schema().schema
        payload = schema

        assert payload["discriminator"]["propertyName"] == "decision"
        assert set(payload["discriminator"]["mapping"]) == {
            "tool_calls",
            "result",
        }
        assert len(payload["oneOf"]) == 2
        assert _decision_branch(schema, "tool_calls")["required"] == [
            "decision",
            "tool_calls",
        ]
        assert _decision_branch(schema, "result")["required"] == [
            "decision",
            "result",
        ]

    def test_build_messages_tells_no_tool_agent_to_return_empty_collection(self):
        strategy = _make_strategy()

        director = strategy._create_director(list[MyIssue], [])
        messages = strategy._build_messages(
            _function_info(return_type=list[MyIssue]),
            [],
            director,
        )

        assert "empty collection instead of null" in str(messages[0].content)
