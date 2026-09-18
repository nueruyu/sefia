from dataclasses import dataclass

import pytest

from sefios import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionConflictError,
    ExecutionManager,
    MemoryPersistence,
    UnknownExecutionError,
)


@dataclass(frozen=True)
class Invocation:
    message: str


@dataclass(frozen=True)
class Result:
    reply: str


async def test_execution_owns_identity_input_state_and_terminal_result():
    manager = ExecutionManager(MemoryPersistence())
    ref = await manager.create(Invocation("hello"), Invocation)

    assert await manager.input(ref, Invocation) == Invocation("hello")

    state = manager.state(ref, "counter", int)
    await state.save(3)
    assert await state.get() == 3

    terminal = await manager.complete(ref, Result("done"), Result)
    assert terminal == ExecutionCompleted(Result("done"))
    assert await manager.terminal(ref, Result) == terminal


async def test_terminal_completion_is_idempotent_and_conflicts_on_change():
    manager = ExecutionManager(MemoryPersistence())
    ref = await manager.create(Invocation("hello"), Invocation)

    await manager.complete(ref, Result("done"), Result)
    assert await manager.complete(ref, Result("done"), Result) == ExecutionCompleted(
        Result("done")
    )

    with pytest.raises(ExecutionConflictError):
        await manager.complete(ref, Result("different"), Result)


async def test_cancel_is_terminal():
    manager = ExecutionManager(MemoryPersistence())
    ref = await manager.create(Invocation("hello"), Invocation)

    assert await manager.cancel(ref) == ExecutionCancelled()
    assert await manager.terminal(ref, Result) == ExecutionCancelled()

    with pytest.raises(ExecutionConflictError):
        await manager.complete(ref, Result("done"), Result)


async def test_unknown_execution_is_rejected():
    from sefios import ExecutionRef

    manager = ExecutionManager(MemoryPersistence())

    with pytest.raises(UnknownExecutionError):
        await manager.input(ExecutionRef("missing"), Invocation)
