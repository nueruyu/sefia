from typing import Annotated, Optional, Protocol

import pytest

from sefia import Tools, infer
from sefia.exceptions import ToolConflictError
from sefia.inference import Capability
from sefia.tool_collectors import DefaultToolCollector


class WebToolkit:
    """A two-method toolkit; discovery only reads its names and docstrings."""

    async def search(self, query: str) -> str:
        """Search the web for a query."""
        raise NotImplementedError

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        raise NotImplementedError


def _collect_self(instance: object, declared: object | None = None):
    """Collect tools with ``instance`` as an @infer method's receiver."""
    return DefaultToolCollector().collect(
        [Capability(value=instance, declared=declared)]
    )


# --------------------------------------------------------------------------- #
# The Tools[...] gate on class-body fields
# --------------------------------------------------------------------------- #


class GreetingToolkit:
    async def greet(self, name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}"

    def _internal_helper(self) -> None: ...


class AppConfig:
    """A held dependency that is not granted — must never leak."""

    def public_setting(self) -> str:
        return "leaked"


class MyAgent:
    _toolkit: Tools[WebToolkit]  # granted, private field
    greeter: Tools[GreetingToolkit]  # granted, public field
    _config: AppConfig  # declared but not granted
    _note: str  # plain data

    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit
        self.greeter = GreetingToolkit()
        self._config = AppConfig()
        self._note = "data"


def test_collect_exposes_only_tools_marked_fields():
    registry = _collect_self(MyAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names
    assert "GreetingToolkit_greet" in tool_names
    assert len(tool_names) == 3
    # A granted toolkit's private method stays internal; ungranted members
    # (config, plain data) expose nothing.
    assert not any("internal_helper" in name for name in tool_names)
    assert not any("setting" in name for name in tool_names)


class UnannotatedFieldAgent:
    """No class-level declaration → fail-closed, nothing discovered."""

    def __init__(self, web: WebToolkit):
        self._web = web


def test_an_undeclared_field_exposes_nothing():
    registry = _collect_self(UnannotatedFieldAgent(WebToolkit()))
    assert registry.get_names() == []


class OptionalAgent:
    _web: Optional[Tools[WebToolkit]]

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_unwraps_optional_around_the_alias():
    registry = _collect_self(OptionalAgent(WebToolkit()))
    assert set(registry.get_names()) == {
        "WebToolkit_search",
        "WebToolkit_fetch_content",
    }


class StackedMetadataAgent:
    _web: Annotated[Tools[WebToolkit], "other metadata"]

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_finds_the_marker_under_stacked_annotated_metadata():
    registry = _collect_self(StackedMetadataAgent(WebToolkit()))
    assert set(registry.get_names()) == {
        "WebToolkit_search",
        "WebToolkit_fetch_content",
    }


class ReadOnlyWeb(Protocol):
    async def search(self, q: str) -> list[str]:
        """Search the web."""
        ...


class BroadWebClient:
    async def search(self, q: str) -> list[str]:
        """Search the web and return matching URLs."""
        return [q]

    async def delete_index(self) -> None:
        """A destructive capability the Protocol does not expose."""
        return None


class ProtocolNarrowedAgent:
    _web: Tools[ReadOnlyWeb]  # granted through a pure narrowing protocol

    def __init__(self, web: BroadWebClient):
        self._web = web


def test_collect_narrows_a_field_to_the_pure_protocol_members():
    registry = _collect_self(ProtocolNarrowedAgent(BroadWebClient()))
    assert set(registry.get_names()) == {"ReadOnlyWeb_search"}


async def test_collect_uses_the_protocol_method_docstring_and_dispatches_to_impl():
    registry = _collect_self(ProtocolNarrowedAgent(BroadWebClient()))
    tool = registry.get("ReadOnlyWeb_search")
    assert tool is not None
    # The Protocol's own docstring is used, not the implementation's.
    assert tool.definition().description == "Search the web."
    # Invocation still dispatches to the concrete implementation.
    assert await tool.invoke({"q": "sefia"}) == ["sefia"]


class BuiltinMarkedAgent:
    _name: Tools[str]  # granting a builtin is meaningless and must not leak

    def __init__(self):
        self._name = "just a string"


def test_a_tools_marked_builtin_exposes_nothing():
    registry = _collect_self(BuiltinMarkedAgent())
    assert registry.get_names() == []


class ConflictingAgent:
    _kit1: Tools[WebToolkit]
    _kit2: Tools[WebToolkit]  # same tool names → conflict

    def __init__(self):
        self._kit1 = WebToolkit()
        self._kit2 = WebToolkit()


def test_collect_tools_with_conflict_raises_error():
    with pytest.raises(ToolConflictError):
        _collect_self(ConflictingAgent())


class ToolkitWithProperty:
    @property
    def status(self) -> str:
        """A property must never be treated as a tool method."""
        raise AssertionError("the property getter must not run during discovery")

    async def check(self) -> str:
        """A real tool method."""
        return "ok"


class PropertyAgent:
    _toolkit: Tools[ToolkitWithProperty]

    def __init__(self):
        self._toolkit = ToolkitWithProperty()


def test_collect_excludes_properties_without_invoking_them():
    registry = _collect_self(PropertyAgent())
    assert set(registry.get_names()) == {"ToolkitWithProperty_check"}


class SlottedAgent:
    __slots__ = ("_toolkit",)
    _toolkit: Tools[WebToolkit]

    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit


def test_collect_finds_dependencies_in_slots():
    registry = _collect_self(SlottedAgent(WebToolkit()))
    tool_names = set(registry.get_names())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names


class SelfMethodAgent:
    """The receiver's own methods are never tools without a surface."""

    async def public_method(self, value: int) -> int:
        """A public helper, but it belongs to the running instance."""
        return value

    @infer
    async def run(self, task: str) -> str:
        """An inference entry point — also never a self-tool."""
        ...


def test_collect_never_exposes_the_instance_own_methods():
    registry = _collect_self(SelfMethodAgent())
    assert registry.get_names() == []


class SubAgent:
    """A nested agent whose @infer method is a tool once granted as a field."""

    @infer
    async def analyze(self, topic: str) -> str:
        """Analyze the topic."""
        ...

    def _helper(self) -> None: ...


class OuterAgent:
    _sub: Tools[SubAgent]

    def __init__(self, sub: SubAgent):
        self._sub = sub


def test_collect_exposes_a_granted_agent_infer_method_as_a_tool():
    registry = _collect_self(OuterAgent(SubAgent()))
    tool_names = set(registry.get_names())
    assert "SubAgent_analyze" in tool_names
    assert not any("_helper" in name for name in tool_names)
    assert len(tool_names) == 1


class OuterHoldingUngranted:
    _sub: SubAgent  # declared but not granted

    def __init__(self):
        self._sub = SubAgent()


def test_a_held_agent_without_the_grant_is_not_exposed():
    registry = _collect_self(OuterHoldingUngranted())
    assert registry.get_names() == []
