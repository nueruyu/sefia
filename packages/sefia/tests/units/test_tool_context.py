import asyncio

import pytest
from sefia import current_tool_call_id, current_tool_call_id_for
from sefia._tool_context import serving_tool_call


def test_raises_outside_a_tool_call():
    with pytest.raises(RuntimeError):
        current_tool_call_id()


def test_reports_the_bound_call_id():
    with serving_tool_call("call-1"):
        assert current_tool_call_id() == "call-1"


def test_nested_binding_restores_the_outer_id():
    # A tool that itself drives an @infer run re-enters the dispatch path, so a
    # binding must restore the caller's id, not clear it.
    with serving_tool_call("outer"):
        with serving_tool_call("inner"):
            assert current_tool_call_id() == "inner"
        assert current_tool_call_id() == "outer"


def test_reports_id_only_for_the_dispatched_function():
    def dispatched():
        pass

    def nested():
        pass

    with serving_tool_call("call-1", dispatched):
        assert current_tool_call_id_for(dispatched) == "call-1"
        assert current_tool_call_id_for(nested) is None

    assert current_tool_call_id_for(dispatched) is None


def test_matches_bound_methods_by_underlying_function():
    class Tool:
        def run(self):
            pass

    tool = Tool()
    with serving_tool_call("call-1", tool.run):
        assert current_tool_call_id_for(tool.run) == "call-1"


async def test_spawned_task_inherits_call_id_after_parent_unbinds() -> None:
    release = asyncio.Event()

    async def background() -> str:
        await release.wait()
        return current_tool_call_id()

    with serving_tool_call("bg-call"):
        task = asyncio.create_task(background())
    try:
        with pytest.raises(RuntimeError):
            current_tool_call_id()
        release.set()
        assert await asyncio.wait_for(task, timeout=1) == "bg-call"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
