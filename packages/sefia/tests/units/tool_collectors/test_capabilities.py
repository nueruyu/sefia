from typing import Protocol


from sefia import Tools
from sefia.inference import Capability, FunctionInfo


class WebToolkit:
    """A two-method toolkit; discovery only reads its names and docstrings."""

    async def search(self, query: str) -> str:
        """Search the web for a query."""
        raise NotImplementedError

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        raise NotImplementedError


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
