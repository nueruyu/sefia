from sefia.llm.step_decision import StepTool, ToolSchemaSource

from ._value_format import StructuredValueFormat


def tool_arguments_format(tool: StepTool) -> StructuredValueFormat:
    if tool.schema_source is ToolSchemaSource.GENERATED:
        return StructuredValueFormat.from_generated_schema(tool.arguments)
    return StructuredValueFormat.from_user_schema(tool.arguments)


__all__ = ["tool_arguments_format"]
