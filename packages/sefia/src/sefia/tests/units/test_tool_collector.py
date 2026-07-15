from typing import Annotated, Any, Protocol

import pytest

from sefia import infer
from sefia.exceptions import ToolConflictError
from sefia.pydantic import PydanticModelBackend
from sefia.tool_collectors import DefaultToolCollector


class WebToolkit:
    """A two-method toolkit; discovery only reads its names and docstrings."""

    async def search(self, query: str) -> str:
        """Search the web for a query."""
        raise NotImplementedError

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        raise NotImplementedError


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
    # __qualname__ includes enclosing scope; verify it ends with the expected suffix
    assert name.endswith("Outer_Inner_my_method")
    # verify no dots or other unsafe characters remain
    assert "." not in name
    assert "<" not in name


def test_schema_builder_caches_results():
    backend = PydanticModelBackend()
    definition1 = backend.definition(example_func, name="example_func")
    definition2 = backend.definition(example_func, name="example_func")
    assert definition1 is definition2


class GreetingToolkit:
    async def greet(self, name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}"

    def _internal_helper(self) -> None: ...


class MyAgent:
    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit  # private member → its public methods are scanned
        self.greeter = GreetingToolkit()  # public member → also scanned


def test_collect_tools_from_public_and_private_members():
    agent = MyAgent(WebToolkit())
    collector = DefaultToolCollector()
    registry = collector.collect(agent)

    tool_names = set(registry.get_names())
    # Held in a private attribute.
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names
    # Held in a public attribute.
    assert "GreetingToolkit_greet" in tool_names
    assert len(tool_names) == 3
    # The private method of a held dependency stays internal.
    assert not any("internal_helper" in name for name in tool_names)


class ConflictingAgent:
    def __init__(self):
        self._kit1 = WebToolkit()
        self._kit2 = WebToolkit()  # same tool names → conflict


def test_collect_tools_with_conflict_raises_error():
    agent = ConflictingAgent()
    collector = DefaultToolCollector()
    with pytest.raises(ToolConflictError):
        collector.collect(agent)


class SelfMethodAgent:
    """A class's own methods are never its own tools — public or private."""

    def __init__(self):
        self._note = "held primitive, not a tool provider"

    async def public_method(self, value: int) -> int:
        """A public helper, but it belongs to the running instance."""
        return value

    async def _private_method(self, value: int) -> int:
        """A private helper."""
        return value

    @infer
    async def run(self, task: str) -> str:
        """An inference entry point — also never a self-tool."""
        ...


def test_collect_never_exposes_the_instance_own_methods():
    collector = DefaultToolCollector()
    registry = collector.collect(SelfMethodAgent())

    # Nothing is collected: the instance holds no dependencies, and its own
    # methods (public, private, or @infer) are never offered back to itself.
    assert registry.get_names() == []


class SubAgent:
    """A nested agent whose @infer method is a tool once held as a dependency."""

    @infer
    async def analyze(self, topic: str) -> str:
        """Analyze the topic."""
        ...

    def _helper(self) -> None: ...


class OuterAgent:
    def __init__(self, sub: SubAgent):
        self._sub = sub  # held agent → its public methods, incl. @infer, are tools


def test_collect_exposes_a_held_agent_infer_method_as_a_tool():
    registry = DefaultToolCollector().collect(OuterAgent(SubAgent()))

    tool_names = set(registry.get_names())
    assert "SubAgent_analyze" in tool_names
    assert not any("_helper" in name for name in tool_names)
    assert len(tool_names) == 1


class ExternalLikeClient:
    """Simulates a third-party class we cannot modify."""

    async def fetch(self, url: str) -> str:
        """Fetch a URL."""
        return url

    def _internal(self) -> None: ...


class RuntimeFallbackAgent:
    """No class-level annotation on the field → falls back to the runtime type."""

    def __init__(self, client):
        self._client = client


def test_collect_falls_back_to_runtime_type_without_class_level_annotation():
    registry = DefaultToolCollector().collect(
        RuntimeFallbackAgent(ExternalLikeClient())
    )

    tool_names = set(registry.get_names())
    assert "ExternalLikeClient_fetch" in tool_names
    assert not any("internal" in name for name in tool_names)
    assert len(tool_names) == 1


class ReadOnlyWeb(Protocol):
    async def search(self, q: str) -> list[str]:
        """Search the web."""
        ...


class ProtocolNarrowedAgent:
    _web: ReadOnlyWeb  # class-level annotation → only the Protocol's members

    def __init__(self, web: "BroadWebClient"):
        self._web = web


class BroadWebClient:
    async def search(self, q: str) -> list[str]:
        """Search the web and return matching URLs."""
        return [q]

    async def delete_index(self) -> None:
        """A destructive capability the Protocol does not expose."""
        return None


def test_collect_narrows_a_protocol_annotated_field_to_its_declared_members():
    registry = DefaultToolCollector().collect(ProtocolNarrowedAgent(BroadWebClient()))

    tool_names = set(registry.get_names())
    assert tool_names == {"ReadOnlyWeb_search"}


async def test_collect_uses_the_protocol_method_docstring_for_the_schema():
    registry = DefaultToolCollector().collect(ProtocolNarrowedAgent(BroadWebClient()))
    tool_info = registry.get("ReadOnlyWeb_search")
    assert tool_info is not None

    # The Protocol's own docstring is used, not the implementation's.
    assert tool_info.definition().description == "Search the web."

    # Invocation still dispatches to the concrete implementation.
    result = await tool_info.invoke({"q": "sefia"})
    assert result == ["sefia"]


class ConcreteAnnotatedAgent:
    _web: WebToolkit  # class-level annotation to a concrete class

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_uses_a_concrete_class_level_annotation():
    registry = DefaultToolCollector().collect(ConcreteAnnotatedAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}


class OptionalAnnotatedAgent:
    _web: WebToolkit | None

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_unwraps_an_optional_class_level_annotation():
    registry = DefaultToolCollector().collect(OptionalAnnotatedAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}


class PrimitiveHoldingAgent:
    def __init__(self):
        self._name = "just a string"
        self._items = [1, 2, 3]
        self._nothing = None


def test_collect_skips_builtin_primitives():
    registry = DefaultToolCollector().collect(PrimitiveHoldingAgent())
    assert registry.get_names() == []


class ToolkitWithProperty:
    @property
    def status(self) -> str:
        """A property must never be treated as a tool method."""
        raise AssertionError("the property getter must not run during discovery")

    async def check(self) -> str:
        """A real tool method."""
        return "ok"


class PropertyAgent:
    def __init__(self):
        self._toolkit = ToolkitWithProperty()


def test_collect_excludes_properties_without_invoking_them():
    registry = DefaultToolCollector().collect(PropertyAgent())

    tool_names = set(registry.get_names())
    assert tool_names == {"ToolkitWithProperty_check"}


class SlottedAgent:
    __slots__ = ("_toolkit",)

    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit


def test_collect_finds_dependencies_in_slots():
    # A slotted agent has no __dict__; its dependency must still be found.
    agent = SlottedAgent(WebToolkit())
    registry = DefaultToolCollector().collect(agent)

    tool_names = set(registry.get_names())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names


class AnyAnnotatedAgent:
    _web: Any  # a common escape-hatch annotation, not a usable interface

    def __init__(self, web: WebToolkit):
        self._web = web


class ObjectAnnotatedAgent:
    _web: object  # same as Any: no usable interface, must not resolve to it

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_falls_back_to_runtime_type_for_any_or_object_annotation():
    # inspect.isclass(Any) is True on 3.11+, so Any/object must be treated as
    # "no declared interface" rather than resolved to a type with no public
    # methods (which would silently produce zero tools).
    for agent in (AnyAnnotatedAgent(WebToolkit()), ObjectAnnotatedAgent(WebToolkit())):
        registry = DefaultToolCollector().collect(agent)
        tool_names = set(registry.get_names())
        assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}


class AnnotatedMetadataAgent:
    _web: Annotated[WebToolkit, "some metadata"]  # class-level, with extras

    def __init__(self, web: WebToolkit):
        self._web = web


def test_collect_unwraps_annotated_metadata_to_the_underlying_class():
    registry = DefaultToolCollector().collect(AnnotatedMetadataAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}


class BadForwardRefAgent:
    _unresolvable: "SomeNameThatDoesNotExist"  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
    _web: WebToolkit

    def __init__(self, web: WebToolkit):
        self._unresolvable = None
        self._web = web


def test_collect_tolerates_an_unresolvable_forward_ref_on_another_field():
    # A NameError from one bad annotation must not crash discovery for the
    # whole instance; the unresolvable field itself has a None value and is
    # skipped, while a sibling field still resolves normally.
    registry = DefaultToolCollector().collect(BadForwardRefAgent(WebToolkit()))

    tool_names = set(registry.get_names())
    assert tool_names == {"WebToolkit_search", "WebToolkit_fetch_content"}
