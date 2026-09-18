from dataclasses import dataclass

import pytest
from sefios import DurableExecutionStore, ExecutionRef, MemoryPersistence
from sefios.exceptions import (
    ExecutionAlreadyTerminalError,
    ExecutionConflictError,
    UnknownExecutionError,
)


@dataclass(frozen=True)
class Invocation:
    message: str


@dataclass(frozen=True)
class Terminal:
    reply: str


def store() -> DurableExecutionStore[Invocation, Terminal]:
    return DurableExecutionStore(
        MemoryPersistence(),
        input_type=Invocation,
        result_type=Terminal,
    )


async def test_execution_owns_immutable_input_and_terminal_result():
    executions = store()
    ref = await executions.create(Invocation("hello"))

    assert await executions.input(ref) == Invocation("hello")
    assert (await executions.snapshot(ref)).status == "active"

    terminal = await executions.complete(ref, Terminal("done"))

    assert terminal == Terminal("done")
    snapshot = await executions.snapshot(ref)
    assert snapshot.status == "completed"
    assert snapshot.result == terminal


async def test_initialization_is_idempotent_but_input_cannot_change():
    executions = store()
    ref = ExecutionRef.new()

    await executions.initialize(ref, Invocation("same"))
    await executions.initialize(ref, Invocation("same"))

    with pytest.raises(ExecutionConflictError):
        await executions.initialize(ref, Invocation("different"))


async def test_terminal_result_is_idempotent_but_conflict_is_rejected():
    executions = store()
    ref = await executions.create(Invocation("hello"))

    assert await executions.complete(ref, Terminal("done")) == Terminal("done")
    assert await executions.complete(ref, Terminal("done")) == Terminal("done")

    with pytest.raises(ExecutionConflictError):
        await executions.complete(ref, Terminal("different"))


async def test_cancellation_is_terminal_and_idempotent():
    executions = store()
    ref = await executions.create(Invocation("hello"))

    await executions.cancel(ref, "user")
    await executions.cancel(ref, "user")

    snapshot = await executions.snapshot(ref)
    assert snapshot.status == "cancelled"
    assert snapshot.cancellation_reason == "user"

    with pytest.raises(ExecutionAlreadyTerminalError):
        await executions.complete(ref, Terminal("late"))


async def test_unknown_execution_is_rejected():
    executions = store()

    with pytest.raises(UnknownExecutionError):
        await executions.snapshot(ExecutionRef("missing"))
