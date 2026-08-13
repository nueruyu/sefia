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


def test_domain_requires_explicit_non_empty_execution_names():
    reports = Domain(glyff.Domain("com.example.reports", version="1"))

    try:
        reports.infer(name="")
    except ValueError as error:
        assert "name" in str(error)
    else:
        raise AssertionError("Expected an empty execution name to be rejected.")
