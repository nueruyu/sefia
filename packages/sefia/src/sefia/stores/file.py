import re
from pathlib import Path
from typing import Any

from glyff.interfaces import Serializer
from glyff_file_store import FileClient

from .._interfaces.session_store import SessionStore

_UNSAFE = re.compile(r'[<>:"\\|?*%\x00-\x1f]')


class FileSessionStore(SessionStore):
    def __init__(self, client: FileClient, serializer: Serializer):
        self._client = client
        self._serializer = serializer

    def _key_to_path(self, key: str) -> Path:
        parts = key.split("/")
        safe_parts = []
        for p in parts:
            if not p or p in (".", ".."):
                raise ValueError(f"Invalid key part: {p!r}")
            safe_parts.append(_UNSAFE.sub(lambda m: f"%{ord(m.group()):02X}", p))
        return Path("sefia", "metadata", *safe_parts).with_suffix(".json")

    async def get(self, key: str, type_hint: type) -> Any | None:
        path = self._key_to_path(key)
        data = await self._client.read(path)
        if data:
            return await self._serializer.deserialize(data, type_hint)
        return None

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        path = self._key_to_path(key)
        data = await self._serializer.serialize(value, type_hint)

        async def _write() -> bytes:
            return data

        await self._client.stage_write(path, _write)

    async def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        await self._client.stage_delete(path)
