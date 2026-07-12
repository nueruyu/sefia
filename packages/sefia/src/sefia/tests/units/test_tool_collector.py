from typing import Annotated, Protocol

import pytest

from sefia import Tools, infer
from sefia._tool_system import Capability, capabilities
from sefia.exceptions import ToolConflictError
from sefia.pydantic import PydanticModelBackend
from sefia.tool_collectors import DefaultToolCollector

from ..conftest import WebToolkit


def _collect_self(instance: object):
    """Collect tools with ``instance`` as an @infer method's ``self`` capability."""
    return DefaultToolCollector().collect([Capability(value=instance, declared=None)])


# --------------------------------------------------------------------------- #
# Schema generation (unchanged surface)
# --------------------------------------------------------------------------- #


def example_func(a: int, b: str = "default") -> bool:
    """An example function."""
    return True


def test_create_tool_schema_from_function():
    definition = PydanticModelBackend().definition(example_func, name="example_func")

    assert definition.name == "example_func"
    assert definition.description == "An example function."

    params = definition.parameters
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "string"
    assert params["properties"]["b"]["default"] == "default"
    assert "a" in params["required"]
    assert "b" not in params.get("required", [])


def test_schema_builder_sanitizes_complex_names():
    backend = PydanticModelBackend()

    class Outer:
        class Inner:
            def my_method(self):
                pass

    name = backend.tool_name(Outer.Inner.my_method)
    assert name.endswith("Outer_Inner_my_method")
    assert "." not in name
    assert "<" not in name


def test_schema_builder_caches_results():
    backend = PydanticModelBackend()
    definition1 = backend.definition(example_func, name="example_func")
    definition2 = backend.definition(example_func, name="example_func")
    assert definition1 is definition2


# --------------------------------------------------------------------------- #
# Capability classification
# --------------------------------------------------------------------------- #


class ToolkitA(Tools):
    async def do_a(self) -> str:
        """Do A."""
        return "a"


def test_self_is_a_capability_parameter_by_convention():
    caps = capabilities({"self": object(), "topic": "x"}, {"topic": str})
    names = {c.declared for c in caps}
    # self classified (declared None); topic (plain data) excluded.
    assert len(caps) == 1
    assert names == {None}


def test_a_role_marked_parameter_is_a_capability_parameter():
    kit = ToolkitA()
    caps = capabilities({"kit": kit, "topic": "x"}, {"kit": ToolkitA, "topic": str})
    assert [c.value for c in caps] == [kit]


def test_a_plain_typed_parameter_is_task_data_not_a_capability():
    caps = capabilities({"topic": "x", "count": 3}, {"topic": str, "count": int})
    assert caps == []


# --------------------------------------------------------------------------- #
# The Tools gate — held fields
# --------------------------------------------------------------------------- #


class GreetingToolkit(Tools):
    async def greet(self, name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}"

    def _internal_helper(self) -> None: ...


class MyAgent:
    _toolkit: WebToolkit  # private, Tools-bearing → scanned
    greeter: GreetingToolkit  # public, Tools-bearing → scanned

    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit
        self.greeter = GreetingToolkit()


def test_collect_tools_from_public_and_private_role_marked_members():
    registry = _collect_self(MyAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names
    assert "GreetingToolkit_greet" in tool_names
    assert len(tool_names) == 3
    # A held toolkit's own private method stays internal.
    assert not any("internal_helper" in name for name in tool_names)


class AppConfig:
    """A held dependency that is not a toolkit — must never leak."""

    def public_setting(self) -> str:
        return "leaked"


class AgentWithNonToolMember:
    _web: WebToolkit
    _config: AppConfig  # not Tools → gated out

    def __init__(self):
        self._web = WebToolkit()
        self._config = AppConfig()


def test_a_non_tools_held_member_is_gated_out():
    registry = _collect_self(AgentWithNonToolMember())

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}
    assert not any("setting" in name for name in tool_names)


class UnannotatedFieldAgent:
    """No class-level annotation → fail-closed, nothing discovered."""

    def __init__(self, web: WebToolkit):
        self._web = web  # held, but undeclared at class level


def test_an_undeclared_field_exposes_nothing_fail_closed():
    registry = _collect_self(UnannotatedFieldAgent(WebToolkit()))
    assert registry.get_names() == []


# --------------------------------------------------------------------------- #
# Plain functions get tools through a role-marked parameter
# --------------------------------------------------------------------------- #


def test_a_plain_function_gets_tools_from_a_role_marked_parameter():
    kit = WebToolkit()
    caps = capabilities({"kit": kit, "topic": "x"}, {"kit": WebToolkit, "topic": str})
    registry = DefaultToolCollector().collect(caps)

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}


def test_annotated_use_site_marker_gates_a_field_in():
    class VendorClient:  # third-party-like: does not inherit Tools
        async def fetch(self, url: str) -> str:
            """Fetch a URL."""
            return url

    class Agent:
        _client: Annotated[VendorClient, Tools]

        def __init__(self):
            self._client = VendorClient()

    registry = _collect_self(Agent())
    names = registry.get_names()
    assert len(names) == 1
    assert names[0].endswith("VendorClient_fetch")


# --------------------------------------------------------------------------- #
# Surface protocol on `self`
# --------------------------------------------------------------------------- #


class ReadOnlyWeb(Tools, Protocol):
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
    _web: ReadOnlyWeb  # declared as the narrowing protocol

    def __init__(self, web: BroadWebClient):
        self._web = web


def test_collect_narrows_a_protocol_annotated_field_to_its_declared_members():
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


class ResearchSurface(Tools, Protocol):
    """Surface that opts an instance's own private method in, and re-narrows a
    held field via a read-only property (the invariance-safe form)."""

    @property
    def _web(self) -> ReadOnlyWeb: ...

    async def _score(self, url: str) -> float: ...


class Researcher:
    _web: BroadWebClient

    def __init__(self):
        self._web = BroadWebClient()

    async def _score(self, url: str) -> float:
        """Score a URL."""
        return 1.0

    @infer
    async def run(self, topic: str) -> str:
        """Research the topic."""
        ...


def test_surface_protocol_opts_in_a_private_method_and_renarrows_a_field():
    registry = DefaultToolCollector().collect(
        [Capability(value=Researcher(), declared=ResearchSurface)]
    )
    tool_names = set(registry.get_names())
    # tier 0: the instance's own _score (declared by the surface protocol)
    assert "ResearchSurface__score" in tool_names
    # tier 1: _web re-narrowed to ReadOnlyWeb, so only search (not delete_index)
    assert "ReadOnlyWeb_search" in tool_names
    assert not any("delete_index" in name for name in tool_names)
    assert len(tool_names) == 2


# --------------------------------------------------------------------------- #
# Concrete annotations, Optional, Annotated-with-extras
# --------------------------------------------------------------------------- #


class ConcreteAnnotatedAgent:
    _web: WebToolkit

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_uses_a_concrete_class_level_annotation():
    registry = _collect_self(ConcreteAnnotatedAgent(WebToolkit()))
    assert set(registry.get_names()) == {"WebToolkit_search", "WebToolkit_fetch_content"}


class OptionalAnnotatedAgent:
    _web: WebToolkit | None

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_unwraps_an_optional_class_level_annotation():
    registry = _collect_self(OptionalAnnotatedAgent(WebToolkit()))
    assert set(registry.get_names()) == {"WebToolkit_search", "WebToolkit_fetch_content"}


class AnnotatedMetadataAgent:
    _web: Annotated[WebToolkit, "some metadata"]

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_unwraps_annotated_metadata_to_the_underlying_class():
    registry = _collect_self(AnnotatedMetadataAgent(WebToolkit()))
    assert set(registry.get_names()) == {"WebToolkit_search", "WebToolkit_fetch_content"}


# --------------------------------------------------------------------------- #
# Non-tool held values, properties, slots, conflicts, self-recursion
# --------------------------------------------------------------------------- #


class PrimitiveHoldingAgent:
    _name: str
    _items: list

    def __init__(self):
        self._name = "just a string"
        self._items = [1, 2, 3]


def test_collect_skips_non_tool_primitives():
    registry = _collect_self(PrimitiveHoldingAgent())
    assert registry.get_names() == []


class ToolkitWithProperty(Tools):
    @property
    def status(self) -> str:
        """A property must never be treated as a tool method."""
        raise AssertionError("the property getter must not run during discovery")

    async def check(self) -> str:
        """A real tool method."""
        return "ok"


class PropertyAgent:
    _toolkit: ToolkitWithProperty

    def __init__(self):
        self._toolkit = ToolkitWithProperty()


def test_collect_excludes_properties_without_invoking_them():
    registry = _collect_self(PropertyAgent())
    assert set(registry.get_names()) == {"ToolkitWithProperty_check"}


class SlottedAgent:
    __slots__ = ("_toolkit",)
    _toolkit: WebToolkit

    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit


def test_collect_finds_dependencies_in_slots():
    registry = _collect_self(SlottedAgent(WebToolkit()))
    tool_names = set(registry.get_names())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names


class ConflictingAgent:
    _kit1: WebToolkit
    _kit2: WebToolkit  # same tool names → conflict

    def __init__(self):
        self._kit1 = WebToolkit()
        self._kit2 = WebToolkit()


def test_collect_tools_with_conflict_raises_error():
    with pytest.raises(ToolConflictError):
        _collect_self(ConflictingAgent())


class SelfMethodAgent:
    """A plain service does not bear Tools, so its own methods — public,
    private, or @infer — are never offered back to itself."""

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


class SubAgent(Tools):
    """A nested agent whose @infer method is a tool once held as a dependency."""

    @infer
    async def analyze(self, topic: str) -> str:
        """Analyze the topic."""
        ...

    def _helper(self) -> None: ...


class OuterAgent:
    _sub: SubAgent  # held agent bearing Tools → its public methods are tools

    def __init__(self, sub: SubAgent):
        self._sub = sub


def test_collect_exposes_a_held_agent_infer_method_as_a_tool():
    registry = _collect_self(OuterAgent(SubAgent()))
    tool_names = set(registry.get_names())
    assert "SubAgent_analyze" in tool_names
    assert not any("_helper" in name for name in tool_names)
    assert len(tool_names) == 1


class UnmarkedSubAgent:
    """A held object that does NOT bear Tools is gated out entirely."""

    @infer
    async def analyze(self, topic: str) -> str:
        """Analyze the topic."""
        ...


class OuterHoldingUnmarked:
    _sub: UnmarkedSubAgent

    def __init__(self):
        self._sub = UnmarkedSubAgent()


def test_a_held_object_without_tools_is_not_exposed():
    registry = _collect_self(OuterHoldingUnmarked())
    assert registry.get_names() == []
