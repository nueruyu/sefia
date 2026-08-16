from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, cast

import jsonschema
import jsonschema.validators
from pydantic import BeforeValidator, WithJsonSchema
from typing_extensions import final


class ToolSchemaKind(Enum):
    TYPED = "typed"
    RAW = "raw"


@final
@dataclass(frozen=True)
class ToolArgumentContract:
    """Original validation schema and its provider-composition policy."""

    schema: dict[str, Any]
    kind: ToolSchemaKind

    def validation_type(self) -> Any:
        validator_cls = jsonschema.validators.validator_for(
            self.schema, default=jsonschema.Draft202012Validator
        )
        validator_cls.check_schema(self.schema)
        validator = validator_cls(self.schema)

        def validate(value: object) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise ValueError("arguments must be a JSON object")
            value_dict = cast(dict[str, Any], value)
            errors = sorted(
                validator.iter_errors(value_dict), key=lambda error: list(error.path)
            )
            if errors:
                raise ValueError("; ".join(error.message for error in errors))
            return value_dict

        return Annotated[
            dict[str, Any],
            BeforeValidator(validate),
            WithJsonSchema({"type": "object"}),
        ]
