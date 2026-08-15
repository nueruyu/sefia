import inspect
from typing import Any, Callable, cast

# Attribute that holds sefia's per-function metadata dict, and the keys under
# which inference policies and the selected profile live inside it.
METADATA_ATTR = "__sefia_metadata__"
KEY_POLICIES = "policies"
KEY_PROFILE_KEY = "profile_key"


def get_metadata(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Return the sefia metadata dict attached to ``func`` (empty if there is none).

    The lookup unwraps ``functools.wraps`` layers, so it works on a function
    regardless of which decorators wrap it.
    """
    return cast(dict[str, Any], getattr(inspect.unwrap(func), METADATA_ATTR, {}))


def ensure_metadata(func: Callable[..., Any]) -> dict[str, Any]:
    """Return the attached metadata dict, creating it on the unwrapped function."""
    underlying = inspect.unwrap(func)
    metadata = getattr(underlying, METADATA_ATTR, None)
    if metadata is None:
        metadata = {}
        setattr(underlying, METADATA_ATTR, metadata)
    return cast(dict[str, Any], metadata)
