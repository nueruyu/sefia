from sefia.llm._tool_call_ids import ToolCallIdRegistry


def test_tool_call_id_registry_reuses_ids_by_index():
    registry = ToolCallIdRegistry()

    first = registry.get_or_create(0)

    assert registry.get_or_create(0) == first
    assert registry.get_or_create(1) != first
