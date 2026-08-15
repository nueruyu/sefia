from collections.abc import Hashable, Sequence

import glyff  # pyright: ignore[reportMissingTypeStubs]
import sefia


def domain(
    id: str,
    *,
    version: str = "1",
    profile: Hashable | None = None,
    policies: Sequence[sefia.Policy] = (),
) -> sefia.Domain:
    """Create an application domain, starting at version 1 by default."""
    return sefia.Domain(
        glyff.Domain(id, version=version),
        default_profile=profile,
        policies=policies,
    )
