import asyncio
import os
import re
from pathlib import Path
from typing import Any

from glyff import Serializer
from typing_extensions import final, override

from ._base import SessionStorage

_UNSAFE = re.compile(r'[<>:"\\|?*%' + r"\x00-\x1f]")
_ESCAPE = re.compile(r"%([0-9A-F]{2})")


@final
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
        safe_parts: list[str] = []
        for part in parts:
            if not part or part in (".", ".."):
                raise ValueError(f"Invalid key part: {part!r}")
            safe_parts.append(
                _UNSAFE.sub(lambda match: f"%{ord(match.group()):02X}", part)
            )
        safe_parts[-1] += ".json"
        return self._base_dir.joinpath(*safe_parts)

    def _path_to_key(self, path: Path) -> str:
        relative = path.relative_to(self._base_dir)
        parts = list(relative.parts)
        if not parts or not parts[-1].endswith(".json"):
            raise ValueError(f"Invalid session-state path: {path}")
        parts[-1] = parts[-1][: -len(".json")]
        decoded = [
            _ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), part)
            for part in parts
        ]
        return "/".join(decoded)

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

    @staticmethod
    def _write_bytes_if_absent(path: Path, data: bytes) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return False
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return True

    def _keys(self, prefix: str) -> tuple[str, ...]:
        if not self._base_dir.exists():
            return ()
        keys = [
            self._path_to_key(path)
            for path in self._base_dir.rglob("*.json")
            if path.is_file()
        ]
        return tuple(sorted(key for key in keys if key.startswith(prefix)))

    @override
    async def get(self, key: str, type_hint: type) -> Any | None:
        path = self._key_to_path(key)
        data = await asyncio.to_thread(self._read_bytes, path)
        if data:
            return await self._serializer.deserialize(data, type_hint)
        return None

    @override
    async def set(self, key: str, value: Any, type_hint: type) -> None:
        path = self._key_to_path(key)
        data = await self._serializer.serialize(value, type_hint)
        await asyncio.to_thread(self._write_bytes, path, data)

    @override
    async def set_if_absent(self, key: str, value: Any, type_hint: type) -> bool:
        path = self._key_to_path(key)
        data = await self._serializer.serialize(value, type_hint)
        return await asyncio.to_thread(self._write_bytes_if_absent, path, data)

    @override
    async def keys(self, prefix: str = "") -> tuple[str, ...]:
        return await asyncio.to_thread(self._keys, prefix)

    @override
    async def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        await asyncio.to_thread(path.unlink, missing_ok=True)
