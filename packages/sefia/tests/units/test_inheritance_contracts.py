from sefia import Domain, HistoryStorage, InferenceStrategy, Session, ToolCollector
from sefia.history_storages import GlyffHistoryStorage
from sefia.llm import LLMInferenceStrategy
from sefia.llm._execution_directors import ExecutionDirector
from sefia.tool_collectors import DefaultToolCollector


def test_extension_points_remain_open():
    assert not getattr(InferenceStrategy, "__final__", False)
    assert not getattr(HistoryStorage, "__final__", False)
    assert not getattr(ToolCollector, "__final__", False)


def test_standard_implementations_are_final():
    assert getattr(LLMInferenceStrategy, "__final__", False)
    assert getattr(GlyffHistoryStorage, "__final__", False)
    assert getattr(DefaultToolCollector, "__final__", False)
    assert getattr(Session, "__final__", False)
    assert getattr(Domain, "__final__", False)


def test_execution_director_template_methods_are_final():
    assert getattr(ExecutionDirector.build_decision_schema, "__final__", False)
    assert getattr(ExecutionDirector.process_response_data, "__final__", False)
