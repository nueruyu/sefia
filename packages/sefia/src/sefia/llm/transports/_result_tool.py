from ..json_schema import JsonSchemaDocument
from ..step_decision import DecisionSpec, StepTool, ToolSchemaSource

_BASE_NAME = "return_result"


def create_result_tool(spec: DecisionSpec) -> StepTool | None:
    if spec.result is None:
        return None
    return StepTool(
        name=_available_name({tool.name for tool in spec.tools}),
        description="Return the final result when the task is complete.",
        arguments=JsonSchemaDocument.from_mapping(
            {
                "type": "object",
                "properties": {"result": spec.result.schema.to_dict()},
                "required": ["result"],
                "additionalProperties": False,
            }
        ),
        schema_source=ToolSchemaSource.GENERATED,
    )


def _available_name(existing: set[str]) -> str:
    name = _BASE_NAME
    suffix = 2
    while name in existing:
        name = f"{_BASE_NAME}_{suffix}"
        suffix += 1
    return name


__all__ = ["create_result_tool"]
