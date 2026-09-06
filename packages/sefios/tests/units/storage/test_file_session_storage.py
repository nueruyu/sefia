from pathlib import Path

import pytest
from glyff_pydantic import PydanticSerializer

from sefios.storage import FileSessionStorage


async def test_keys_with_dotted_tails_do_not_collide(tmp_path: Path) -> None:
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())

    await store.set("state.v1", {"value": "one"}, dict)
    await store.set("state.v2", {"value": "two"}, dict)

    assert await store.get("state.v1", dict) == {"value": "one"}
    assert await store.get("state.v2", dict) == {"value": "two"}


async def test_rejects_unsafe_key_parts(tmp_path: Path) -> None:
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())

    with pytest.raises(ValueError):
        await store.get("..", dict)
