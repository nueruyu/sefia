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
