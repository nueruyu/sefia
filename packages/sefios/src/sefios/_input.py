"""Application-controlled Input interaction identity."""

import hashlib
import json

from glyff import ExecutionId


def _execution_id_to_data(execution_id: ExecutionId) -> dict[str, object]:
    parent_id = execution_id.parent_id
    return {
        "domain_id": execution_id.domain_id.value,
        "name": execution_id.name.value,
        "sequence": execution_id.sequence,
        "arguments_digest": execution_id.arguments_digest.value,
        "parent_id": _execution_id_to_data(parent_id) if parent_id else None,
    }


def interaction_id_for_execution(execution_id: ExecutionId) -> str:
    """Derive an opaque external-input identity from an engraved execution."""
    data = _execution_id_to_data(execution_id)
    stable_repr = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()
