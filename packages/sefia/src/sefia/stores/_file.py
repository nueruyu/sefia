import os
import re
from pathlib import Path
from typing import Any

from glyff import Serializer

from .._interfaces.session_store import SessionStore

_UNSAFE = re.compile(r'[<>:"\\|?*%' + r"\x00-\x1f]")


class FileSessionStore(SessionStore):
    """A file-based metadata store for session-scoped sefia state.

    Each key maps to a JSON file under ``base_dir``. Writes are committed
    immediately (temp file + atomic rename), so state written before a pause is
    durably persisted and visible when the run resumes.
    """

    def __init__(self, base_dir: str | Path, serializer: Serializer):
        self._base_dir = Path(base_dir)
        self._serializer = serializer

    def _key_to_path(self, key: str) -> Path:
        parts = key.split("/")
        safe_parts = []
        for p in parts:
            if not p or p in (".", ".."):
                raise ValueError(f"Invalid key part: {p!r}")
            safe_parts.append(_UNSAFE.sub(lambda m: f"%{ord(m.group()):02X}", p))
        return self._base_dir.joinpath("sefia", "metadata", *safe_parts).with_suffix(
            ".json"
        )

    async def get(self, key: str, type_hint: type) -> Any | None:
        path = self._key_to_path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        if data:
            return await self._serializer.deserialize(data, type_hint)
        return None

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        path = self._key_to_path(key)
        data = await self._serializer.serialize(value, type_hint)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    async def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
