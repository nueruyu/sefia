import glyff
from glyff.store import MemoryBackend

from sefia import Domain
from sefia.testing import MockLLMClient, memory_session, result_response


async def test_domain_inference_records_stable_application_and_runtime_boundaries():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="2"))

    @reports.infer(name="summarize")
    async def summarize(document: str) -> str: ...

    async with memory_session(
        MockLLMClient([result_response("summary")]),
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
    step = by_name["inference_step"]
    assert outer.id.domain_id == glyff.DomainId("com.example.reports")
    assert step.id.domain_id == glyff.DomainId("sefia.runtime")
    assert step.id.parent_id == outer.id


async def test_domain_infer_uses_the_qualified_function_name():
    backend = MemoryBackend()
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    class Reporter:
        @reports.infer
        async def prepare(self, document: str) -> str: ...

    async with memory_session(
        MockLLMClient([result_response("summary")]),
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
    assert any(
        execution.id.name.value.endswith("Reporter.prepare") for execution in executions
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
    assert executions[0].id.name == glyff.ExecutionName("prepare")


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


def test_domain_requires_explicit_non_empty_execution_names():
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    try:
        reports.infer(name="")
    except ValueError as error:
        assert "name" in str(error)
    else:
        raise AssertionError("Expected an empty execution name to be rejected.")

    try:
        reports.engrave(name="")
    except ValueError as error:
        assert "name" in str(error)
    else:
        raise AssertionError("Expected an empty execution name to be rejected.")
