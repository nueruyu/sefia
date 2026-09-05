from sefia import HistorySnapshot
from sefia._history import StepHistory
from sefia.inference import ToolCallsDecision, ToolCallResult
from sefia.testing import make_tool_call_request


def _decision(i: int) -> ToolCallsDecision:
    return ToolCallsDecision(
        calls=[make_tool_call_request(id=str(i), name="a_tool", arguments={"i": i})]
    )


def _result(i: int) -> ToolCallResult:
    return ToolCallResult(tool_call_id=str(i), result=f"r{i}")


class TestStepHistory:
    def test_starts_from_the_given_items(self):
        history = StepHistory([_decision(0), _result(0)])
        assert list(history.items) == [_decision(0), _result(0)]

    def test_extend_appends(self):
        history = StepHistory()
        history.extend([_decision(0), _result(0)])
        history.extend([_decision(1)])
        assert list(history.items) == [_decision(0), _result(0), _decision(1)]

    def test_rewrite_replaces(self):
        history = StepHistory([_decision(0), _result(0), _decision(1)])
        history.rewrite([_decision(1)])
        assert list(history.items) == [_decision(1)]

    def test_items_is_an_immutable_snapshot(self):
        history = StepHistory()
        history.extend([_decision(0)])

        view = history.items
        assert isinstance(view, tuple)
        history.extend([_result(0)])
        assert len(view) == 1  # the earlier snapshot is unaffected


class TestHistorySnapshot:
    def test_defaults_to_empty(self):
        snap = HistorySnapshot()
        assert snap.items == ()
        assert snap.completed_steps == 0
