"""External input from deterministic application control flow."""

import glyff

from ._glyff import GLYFF_DOMAIN
from ._input import interaction_id_for_execution
from .interactions import require_interaction


@GLYFF_DOMAIN.engrave(name="require_input")
async def require_input(prompt: str) -> str:
    """Request text with an identity derived from this engraved execution."""
    execution_id = glyff.get_context().current_execution_id
    if execution_id is None:
        raise RuntimeError("require_input() requires an engraved execution.")
    return await require_interaction(
        interaction_id_for_execution(execution_id),
        {"type": "input", "prompt": prompt},
        str,
    )
