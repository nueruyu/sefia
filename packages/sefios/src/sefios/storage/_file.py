import asyncio
import os
import re
from pathlib import Path
from typing import Any

from glyff import Serializer

from ._base import SessionStorage

_UNSAFE = re.compile(r'[<>:"\\|?*%' + r"\x00-\x1f]")


class FileSessionStorage(SessionStorage):
    """A file-based storage for session-scoped state.

    Each key maps to a JSON file under ``base_dir``. Writes are committed
    immediately (temp file + atomic rename), so state written before a pause is
    durably persisted and visible when the run resumes. File I/O runs in a
    worker thread so the event loop is never blocked.
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
        # Append (not with_suffix) so keys differing only in a dotted tail
        # ("state.v1" vs "state.v2") cannot collide on the same file.
        safe_parts[-1] += ".json"
        return self._base_dir.joinpath(*safe_parts)

    @staticmethod
    def _read_bytes(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    async def get(self, key: str, type_hint: type) -> Any | None:
        path = self._key_to_path(key)
        data = await asyncio.to_thread(self._read_bytes, path)
        if data:
            return await self._serializer.deserialize(data, type_hint)
        return None

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        path = self._key_to_path(key)
        data = await self._serializer.serialize(value, type_hint)
        await asyncio.to_thread(self._write_bytes, path, data)

    async def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        await asyncio.to_thread(path.unlink, missing_ok=True)
