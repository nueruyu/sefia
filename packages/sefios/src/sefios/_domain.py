from collections.abc import Hashable, Sequence

import glyff
import sefia


def domain(
    id: str,
    *,
    version: str,
    profile: Hashable | None = None,
    policies: Sequence[sefia.Policy] = (),
) -> sefia.Domain:
    """Create an application domain with Sefios authoring defaults."""
    return sefia.Domain(
        glyff.Domain(id, version=version),
        default_profile=profile,
        policies=policies,
    )
