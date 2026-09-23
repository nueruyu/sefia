from sefia._tool_system import ToolRegistry
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.transports._native._result_tool import create_result_tool
from sefia.pydantic import PydanticResultFormatFactory


def test_result_tool_avoids_name_collision() -> None:
    def return_result(value: str) -> str:
        return value

    result_format_factory = PydanticResultFormatFactory()
    registry = ToolRegistry()
    registry.add(return_result, name="return_result")
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=registry.get_all(),
        result_format_factory=result_format_factory,
    )
    result_tool = create_result_tool(decision)
    assert result_tool is not None
    assert result_tool.name == "return_result_2"
