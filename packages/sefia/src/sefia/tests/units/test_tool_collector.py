from typing import Annotated, Optional, Protocol

import pytest

from sefia import Tools, infer
from sefia.exceptions import ToolConflictError
from sefia.inference import Capability, FunctionInfo
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


def _collect_self(instance: object, declared: object | None = None):
    """Collect tools with ``instance`` as an @infer method's receiver."""
    return DefaultToolCollector().collect(
        [Capability(value=instance, declared=declared)]
    )


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
# Capability classification: the receiver, on the call descriptor
# --------------------------------------------------------------------------- #


class SurfaceForInfo(Protocol): ...


class InfoService:
    async def plain(self, topic: str) -> str:
        """No self annotation."""
        ...

    async def surfaced(self: SurfaceForInfo, topic: str) -> str:
        """Self annotated with a surface protocol."""
        ...


def test_the_receiver_is_the_capability_and_the_rest_is_prompt_data():
    svc = InfoService()
    info = FunctionInfo.create(InfoService.plain, (svc, "x"), {})
    assert info.capabilities == [Capability(value=svc, declared=None)]
    assert info.prompt_arguments == {"topic": "x"}


def test_an_annotated_self_carries_its_surface():
    svc = InfoService()
    info = FunctionInfo.create(InfoService.surfaced, (svc, "x"), {})
    assert info.capabilities == [Capability(value=svc, declared=SurfaceForInfo)]


def test_plain_function_parameters_are_never_capabilities():
    async def run(kit: Tools[WebToolkit], topic: str) -> str:
        """A plain function: every parameter is task data."""
        ...

    kit = WebToolkit()
    info = FunctionInfo.create(run, (kit, "x"), {})
    assert info.capabilities == []
    assert info.prompt_arguments == {"kit": kit, "topic": "x"}


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


# --------------------------------------------------------------------------- #
# Surface protocols on `self`
# --------------------------------------------------------------------------- #


class ResearchSurface(Protocol):
    """A pure protocol: annotating ``self`` with it is the whole opt-in.

    Field re-narrowing uses a read-only property (a plain protocol attribute
    is invariant and would not type-check against the concrete field type).
    """

    @property
    def _web(self) -> ReadOnlyWeb: ...

    async def _score(self, url: str) -> float: ...


class Researcher:
    _web: Tools[BroadWebClient]
    _config: AppConfig

    def __init__(self):
        self._web = BroadWebClient()
        self._config = AppConfig()

    async def _score(self, url: str) -> float:
        """Score a URL."""
        return 1.0

    @infer
    async def run(self, topic: str) -> str:
        """Research the topic."""
        ...


def test_a_surface_protocol_replaces_the_class_body_grant():
    registry = _collect_self(Researcher(), declared=ResearchSurface)
    tool_names = set(registry.get_names())
    # The instance's own private method, opted in by the surface declaration.
    assert "ResearchSurface__score" in tool_names
    # The field, re-narrowed by the surface's property to ReadOnlyWeb.
    assert "ReadOnlyWeb_search" in tool_names
    assert not any("delete_index" in name for name in tool_names)
    assert len(tool_names) == 2


class DataBearingSurface(Protocol):
    workspace: str  # a data member: no callable interface, nothing to expose

    async def _score(self, url: str) -> float: ...


class WorkspaceService:
    workspace: str

    def __init__(self):
        self.workspace = "/tmp/w"

    async def _score(self, url: str) -> float:
        return 1.0


def test_a_surface_data_member_exposes_no_tools():
    registry = _collect_self(WorkspaceService(), declared=DataBearingSurface)
    assert set(registry.get_names()) == {"DataBearingSurface__score"}


def test_a_concrete_self_annotation_behaves_like_the_class_body_path():
    # Only protocols act as surfaces; a concrete class annotation on self
    # falls back to scanning that class's Tools-marked fields.
    registry = _collect_self(Researcher(), declared=Researcher)
    tool_names = set(registry.get_names())
    assert not any("_score" in name for name in tool_names)
    assert {"BroadWebClient_search", "BroadWebClient_delete_index"} == tool_names
