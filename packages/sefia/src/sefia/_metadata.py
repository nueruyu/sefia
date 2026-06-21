import inspect
from typing import Callable

# Attribute that holds sefia's per-function metadata dict, and the key under
# which inference policies live inside it.
METADATA_ATTR = "__sefia_metadata__"
POLICIES_KEY = "policies"


def get_metadata(func: Callable) -> dict:
    """
    Return the sefia metadata dict attached to ``func`` (empty if there is none).

    The lookup unwraps ``functools.wraps`` layers, so it works on a function
    regardless of which decorators wrap it.
    """
    return getattr(inspect.unwrap(func), METADATA_ATTR, {})


def ensure_metadata(func: Callable) -> dict:
    """Return the attached metadata dict, creating it on the unwrapped function."""
    underlying = inspect.unwrap(func)
    metadata = getattr(underlying, METADATA_ATTR, None)
    if metadata is None:
        metadata = {}
        setattr(underlying, METADATA_ATTR, metadata)
    return metadata
