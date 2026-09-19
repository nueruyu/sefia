from collections.abc import Awaitable, Callable
from typing import cast

import glyff
import pytest
from typing_extensions import final, override
from glyff.store import MemoryBackend

from sefia import (
    DecisionContext,
    DecisionMiddleware,
    Domain,
    InferenceMiddleware,
    Policy,
    Profile,
    StepMiddleware,
    policy,
)
from sefia.inference import ResultDecision, StepDecision, ToolCallsDecision
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_completion,
)


async def test_domain_inference_records_stable_application_and_runtime_boundaries():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="2"))

    @reports.infer(name="summarize")
    async def summarize(document: str) -> str: ...

    async with memory_session(
        MockLLMClient([result_completion("summary")]),
        session_id="domain-identities",
        backend=backend,
    ):
        assert await summarize("document") == "summary"

    executions = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId("domain-identities")
        )
    ]
    by_name = {execution.id.name.value: execution for execution in executions}

    outer = by_name["summarize"]
    step = by_name["inference.step"]
    assert outer.id.domain_id == glyff.DomainId("com.example.reports")
    assert step.id.domain_id == glyff.DomainId("sefia")
    assert step.id.parent_id == outer.id


async def test_domain_infer_uses_the_qualified_function_name():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    class Reporter:
        @reports.infer
        async def prepare(self, document: str) -> str: ...

    async with memory_session(
        MockLLMClient([result_completion("summary")]),
        session_id="implicit-inference-name",
        backend=backend,
    ):
        assert await Reporter().prepare("document") == "summary"

    executions = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId("implicit-inference-name")
        )
    ]
    application_execution = next(
        execution
        for execution in executions
        if execution.id.domain_id == glyff.DomainId("com.example.reports")
    )
    assert (
        application_execution.id.name.value
        == f"{__name__}.{Reporter.prepare.__qualname__}"
    )


async def test_domain_engrave_uses_the_function_name():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    @reports.engrave
    async def prepare(document: str) -> str:
        return document.upper()

    async with memory_session(
        MockLLMClient([]), session_id="domain-engrave", backend=backend
    ):
        assert await prepare("draft") == "DRAFT"

    executions = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId("domain-engrave")
        )
    ]
    assert len(executions) == 1
    assert executions[0].id.name.value.endswith(
        "test_domain_engrave_uses_the_function_name.<locals>.prepare"
    )


async def test_domain_engrave_accepts_an_explicit_name():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    @reports.engrave(name="prepare_report")
    async def prepare(document: str) -> str:
        return document.upper()

    async with memory_session(
        MockLLMClient([]), session_id="named-domain-engrave", backend=backend
    ):
        assert await prepare("draft") == "DRAFT"

    executions = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId("named-domain-engrave")
        )
    ]
    assert len(executions) == 1
    assert executions[0].id.name == glyff.ExecutionName("prepare_report")


@pytest.mark.parametrize("invalid_return", [False, True])
async def test_rejected_decision_is_uncommitted_and_regenerated_on_resume(
    invalid_return: bool,
) -> None:
    backend = MemoryBackend()
    instances: list[DecisionMiddleware] = []
    observed: list[StepDecision] = []

    @final
    class RejectFirst(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            decision = await nxt()
            observed.append(decision)
            if len(observed) == 1:
                if invalid_return:
                    return cast(StepDecision, "invalid")
                raise ValueError("rejected decision")
            return decision

    def middleware() -> list[InferenceMiddleware | StepMiddleware | DecisionMiddleware]:
        instance = RejectFirst()
        instances.append(instance)
        return [instance]

    reports = Domain(
        glyff.Domain("com.example.reports", version="1"),
        policies=[Policy(middleware=middleware)],
    )

    @reports.infer(name="summarize")
    async def summarize(document: str) -> str: ...

    client = MockLLMClient(
        [result_completion("rejected"), result_completion("accepted")]
    )
    session_id = "decision-resume"
    async with memory_session(client, session_id=session_id, backend=backend):
        with pytest.raises(
            TypeError if invalid_return else ValueError,
            match="Unknown decision type" if invalid_return else "rejected decision",
        ):
            await summarize("document")

    unfinished = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId(session_id)
        )
        if execution.id.name.value == "inference.step"
    ]
    assert len(unfinished) == 1
    assert unfinished[0].status is glyff.ExecutionStatus.STARTED
    assert unfinished[0].result is None

    async with memory_session(client, session_id=session_id, backend=backend):
        assert await summarize("document") == "accepted"

    steps = [
        execution
        async for execution in backend.repository.executions(
            glyff.SessionId(session_id)
        )
        if execution.id.name.value == "inference.step"
    ]
    assert len(steps) == 1
    assert steps[0].id == unfinished[0].id
    assert steps[0].status is glyff.ExecutionStatus.COMPLETED
    assert steps[0].result is not None
    assert observed == [ResultDecision("rejected"), ResultDecision("accepted")]
    assert len(client.requests) == 2
    assert len(instances) == 2
    assert instances[0] is not instances[1]

    async with memory_session(client, session_id=session_id, backend=backend):
        assert await summarize("document") == "accepted"
    assert len(client.requests) == 2
    assert len(observed) == 2


@pytest.mark.parametrize("policy_outside_infer", [False, True])
async def test_decision_middleware_inherits_policy_precedence_and_run_scope(
    policy_outside_infer: bool,
) -> None:
    calls: list[str] = []
    built: list[DecisionMiddleware] = []

    @final
    class Record(DecisionMiddleware):
        def __init__(self, label: str) -> None:
            self.label = label
            self.steps: list[int] = []

        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            self.steps.append(ctx.step)
            assert self.steps == list(range(ctx.step + 1))
            calls.append(f"{self.label}:enter:{ctx.step}")
            decision = await nxt()
            calls.append(f"{self.label}:exit:{ctx.step}")
            if self.label == "function" and ctx.step == 0:
                return ToolCallsDecision([])
            return decision

    def record_policy(label: str) -> Policy:
        def middleware() -> list[
            InferenceMiddleware | StepMiddleware | DecisionMiddleware
        ]:
            instance = Record(label)
            built.append(instance)
            return [instance]

        return Policy(middleware=middleware)

    reports = Domain(
        glyff.Domain("com.example.reports", version="1"),
        default_profile="reporting",
        policies=[record_policy("domain")],
    )

    async def summarize(document: str) -> str: ...

    function_policy = policy(record_policy("function"))
    if policy_outside_infer:
        inferred = function_policy(reports.infer(summarize))
    else:
        inferred = reports.infer(function_policy(summarize))

    client = MockLLMClient(
        [
            result_completion("continue"),
            result_completion("first"),
            result_completion("continue"),
            result_completion("second"),
        ]
    )
    async with memory_session(
        MockLLMClient([]),
        policies=[record_policy("context")],
        profiles=[
            Profile(key="reporting", client=client, policies=[record_policy("profile")])
        ],
    ):
        assert await inferred("one") == "first"
        assert await inferred("two") == "second"

    labels = ["context", "domain", "profile", "function"]
    assert calls == [
        entry
        for _ in range(2)
        for step in (0, 1)
        for entry in (
            *[f"{label}:enter:{step}" for label in labels],
            *[f"{label}:exit:{step}" for label in reversed(labels)],
        )
    ]
    assert len(built) == 8
    assert len({id(instance) for instance in built}) == 8
