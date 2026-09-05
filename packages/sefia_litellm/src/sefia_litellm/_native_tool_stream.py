from dataclasses import dataclass, field
from typing import Protocol

from typing_extensions import final

from sefia.llm.structured_data import StructuredData
from sefia.llm.streaming import (
    JsonOutputStreamDecoder,
    OutputStreamEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._schema._data_format import StructuredDataFormat


class _FunctionCallDelta(Protocol):
    @property
    def name(self) -> str | None: ...

    @property
    def arguments(self) -> str: ...


class NativeToolCallDelta(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def function(self) -> _FunctionCallDelta: ...


@dataclass
class _ToolCallState:
    name: str | None = None
    arguments_json: str = ""
    decoded_length: int = 0
    decoder: JsonOutputStreamDecoder = field(default_factory=JsonOutputStreamDecoder)


@final
class NativeToolCallStreamDecoder:
    """Decodes LiteLLM tool-call fragments into logical decision events."""

    def __init__(self, tool_data_formats: dict[str, StructuredDataFormat]) -> None:
        self._tool_data_formats = tool_data_formats
        self._calls: dict[int, _ToolCallState] = {}

    def feed(self, calls: list[NativeToolCallDelta]) -> list[OutputStreamEvent]:
        events: list[OutputStreamEvent] = []
        for call in calls:
            state = self._calls.setdefault(call.index, _ToolCallState())
            name = call.function.name
            if name and state.name is None:
                state.name = name
                events.append(StringEnd(("tool_calls", call.index, "name"), name))
            if call.function.arguments:
                state.arguments_json += call.function.arguments
            events.extend(self._decode_available(call.index, state))
        return events

    def finish(self) -> list[OutputStreamEvent]:
        events: list[OutputStreamEvent] = []
        for index, state in self._calls.items():
            data_format = (
                self._tool_data_formats.get(state.name)
                if state.name is not None
                else None
            )
            if data_format is None or not data_format.transforms_data:
                events.extend(self._decode_available(index, state))
                continue
            try:
                data = data_format.decode(
                    StructuredData.parse_json(state.arguments_json)
                )
            except ValueError:
                continue
            events.extend(
                _structured_data_events(data, ("tool_calls", index, "arguments"))
            )
        return events

    def _decode_available(
        self, index: int, state: _ToolCallState
    ) -> list[OutputStreamEvent]:
        if state.name is None:
            return []
        data_format = self._tool_data_formats.get(state.name)
        if data_format is not None and data_format.transforms_data:
            return []

        fragment = state.arguments_json[state.decoded_length :]
        if not fragment:
            return []
        state.decoded_length = len(state.arguments_json)
        return [
            _tool_argument_event(index, event) for event in state.decoder.feed(fragment)
        ]


def _structured_data_events(
    data: StructuredData,
    path: tuple[str | int, ...],
) -> list[OutputStreamEvent]:
    tree = data.tree
    if isinstance(tree, str):
        return [StringEnd(path, tree)]
    if tree is None or isinstance(tree, int | float | bool):
        return [Scalar(path, tree)]
    if isinstance(tree, list):
        return [
            event
            for index, item in enumerate(tree)
            for event in _structured_data_events(
                StructuredData.from_tree(item), (*path, index)
            )
        ]
    return [
        event
        for name, item in tree.items()
        for event in _structured_data_events(
            StructuredData.from_tree(item),
            (*path, name if isinstance(name, str | int) else str(name)),
        )
    ]


def _tool_argument_event(index: int, event: OutputStreamEvent) -> OutputStreamEvent:
    path = ("tool_calls", index, "arguments", *event.path)
    if isinstance(event, StringDelta):
        return StringDelta(path, event.text)
    if isinstance(event, StringEnd):
        return StringEnd(path, event.value)
    assert isinstance(event, Scalar)
    return Scalar(path, event.value)
