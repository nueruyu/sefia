from typing import Protocol


from sefia import Tools, infer
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


class AppConfig:
    def public_setting(self) -> str:
        return "leaked"


class ReadOnlyWeb(Protocol):
    async def search(self, q: str) -> list[str]: ...


class BroadWebClient:
    async def search(self, q: str) -> list[str]:
        return [q]

    async def delete_index(self) -> None:
        return None


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
