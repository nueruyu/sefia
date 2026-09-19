"""First-write-wins resolution across independent Python processes."""

import asyncio
import sys
from pathlib import Path

import pytest
from glyff_pydantic import PydanticSerializer
from sefios.interactions import InteractionChannel, InteractionResult
from sefios.storage import FileSessionStorage, SQLiteSessionStorage

_WORKER = """
import asyncio
import sys
from glyff_pydantic import PydanticSerializer
from sefios.exceptions import InteractionConflictError
from sefios.interactions import InteractionChannel
from sefios.storage import FileSessionStorage, SQLiteSessionStorage

backend, path, result = sys.argv[1:]
serializer = PydanticSerializer()
store = (SQLiteSessionStorage(path, "session", serializer) if backend == "sqlite"
         else FileSessionStorage(path, serializer))
channel = InteractionChannel(store)
print("ready", flush=True)
sys.stdin.readline()
try:
    asyncio.run(channel.resolve("shared", int(result)))
except InteractionConflictError:
    print("conflict")
else:
    print("resolved")
"""


@pytest.mark.parametrize("backend", ["sqlite", "file"])
async def test_resolution_across_processes(tmp_path: Path, backend: str) -> None:
    path = tmp_path / "store"
    serializer = PydanticSerializer()
    store = (
        SQLiteSessionStorage(path, "session", serializer)
        if backend == "sqlite"
        else FileSessionStorage(path, serializer)
    )
    channel = InteractionChannel(store)
    await channel.request("shared", {"request": "value"})
    processes: list[asyncio.subprocess.Process] = []
    try:
        for i in range(4):
            processes.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    _WORKER,
                    backend,
                    str(path),
                    str(i),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
        for process in processes:
            assert process.stdout is not None
            assert await asyncio.wait_for(process.stdout.readline(), 30) == b"ready\n"
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(process.communicate(b"resolve\n") for process in processes)
            ),
            30,
        )
        for process, (_, stderr) in zip(processes, outputs):
            assert process.returncode == 0, stderr.decode()
        outcomes = [stdout.strip() for stdout, _ in outputs]
        assert outcomes.count(b"resolved") == 1
        assert outcomes.count(b"conflict") == 3
        assert await channel.request(
            "shared", {"request": "value"}
        ) == InteractionResult(outcomes.index(b"resolved"))
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
