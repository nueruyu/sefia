from copy import deepcopy
from typing import Any, cast

from pydantic import ConfigDict, TypeAdapter, create_model


def build_llm_schema(model: Any, *, name: str) -> dict[str, Any]:
    """Build the provider-facing envelope around a validation model."""
    envelope = create_model(
        f"{name}Envelope",
        __config__=ConfigDict(extra="forbid"),
        payload=(model, ...),
    )
    schema = _normalize_schema(TypeAdapter(envelope).json_schema())
    schema["description"] = "The model for the LLM's decision on the next action."
    return schema


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a closed schema accepted by the supported live providers."""
    result = deepcopy(schema)

    def normalize(node: Any) -> None:
        if isinstance(node, list):
            for item in cast(list[Any], node):
                normalize(item)
            return
        if not isinstance(node, dict):
            return
        node_dict = cast(dict[str, Any], node)

        if node_dict.get("type") == "object":
            node_dict.setdefault("additionalProperties", False)
            properties = node_dict.get("properties")
            if isinstance(properties, dict):
                node_dict["required"] = list(cast(dict[str, Any], properties))

        # OpenAI accepts nested ``anyOf`` but rejects ``oneOf``. The validation
        # model remains unchanged and enforces the original union semantics.
        one_of = node_dict.pop("oneOf", None)
        if one_of is not None:
            node_dict["anyOf"] = one_of

        for value in node_dict.values():
            normalize(value)

    normalize(result)
    return result
