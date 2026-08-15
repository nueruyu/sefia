# Human-in-the-loop without re-running the turn

*The hard part of human-in-the-loop LLM flows isn't calling the model — it's resuming a turn that
paused, without redoing the expensive, side-effecting, or already-approved work.*

## The scenario

A research assistant turn:

1. **clarify** the task — an LLM call
2. **search** the web — a tool call (costs money, rate-limited)
3. **draft** a report — an LLM call (expensive)
4. **pause** for a human to approve the draft
5. **finalize** — send the report (a side effect)

The user approves the draft, then the process restarts — a deploy, a crash, a
scale-down, or simply a new request handling the approval. The turn must continue
from step 5 **without** re-clarifying, re-searching (re-paying), re-drafting, or
re-sending.

## What "resume" actually has to do

- **Skip completed steps** — don't re-call the LLM or the tools.
- **Replay their exact outputs** — not to save cost, but for **correctness**: an
  LLM call is non-deterministic, so re-running `draft` produces a *different*
  draft than the one the human approved.
- **Fire side effects exactly once** — a crash between "sent" and "recorded sent"
  must not send twice.
- **Survive a process restart** — the answer may arrive in a fresh process.
- **Stay consistent under concurrent re-entry** — two requests resuming the same
  session must not double-run.

## Hand-rolled

A competent first attempt. Note how much of it is bookkeeping, not logic:

```python
async def research_turn(session_id: str, task: str, approval: str | None = None):
    state = await db.load(session_id) or {"results": {}}
    r = state["results"]

    # 1. clarify (LLM) — MUST cache. Re-running yields a different brief,
    #    which would invalidate the search and draft below.
    if "clarify" not in r:
        r["clarify"] = await llm_clarify(task)
        await db.save(session_id, state)          # persist before the next step
    brief = r["clarify"]

    # 2. search (paid, rate-limited) — MUST NOT repeat on resume.
    if "search" not in r:
        r["search"] = await web_search(brief)
        await db.save(session_id, state)
    sources = r["search"]

    # 3. draft (expensive LLM) — MUST cache. The human approves THIS draft;
    #    re-running would produce a different one and the approval would be stale.
    if "draft" not in r:
        r["draft"] = await llm_draft(brief, sources)
        await db.save(session_id, state)
    draft = r["draft"]

    # 4. pause for approval.
    if "approved" not in r:
        if approval is None:
            state["awaiting"] = "approval"
            await db.save(session_id, state)
            raise Paused(session_id, draft)        # handler returns 202 + draft
        r["approved"] = approval
        await db.save(session_id, state)

    # 5. finalize (side effect). A crash AFTER send but BEFORE save re-runs this
    #    step → must be idempotent, keyed so the provider dedupes.
    if "finalize" not in r:
        await send_report(draft, idempotency_key=f"{session_id}:finalize")
        r["finalize"] = True
        await db.save(session_id, state)

    return draft
```

### The traps hiding in that code

- **Cache for correctness, not just cost** — forget to cache `draft` and a resume
  regenerates it; the human approved a draft that no longer exists.
- **Persist after every step**, and give every side-effecting step its own
  idempotency key (step 5).
- **Step keys are brittle** — reorder or insert a step and progress maps to the wrong
  slot. Concurrent re-entry needs a lock you haven't written.
- **It only grows** — add a tool, a loop, or a retry and the `if "x" not in r`
  bookkeeping multiplies. You are maintaining a small, bespoke resume engine.

## With sefia

The turn is an ordinary typed function; the pause is a tool that raises:

```python
from sefios import SQLitePersistence, domain
from sefios.fastapi import SefiaHTTP
from sefios.fastapi.exceptions import InputRequired
from sefios.tools import Input

infer = domain("research").infer


class Research:
    def __init__(self, web: WebToolkit, input_tool: Input):
        self._web = web
        self._input = input_tool

    @infer
    async def run(self, task: str) -> Report:
        """Clarify the task, research it with web search, draft a report, get the
        human's approval of the draft, then finalize and send."""
        ...


api = SefiaHTTP(
    model="gpt-4o",
    persistence=SQLitePersistence(),
)
service = Research(web=WebToolkit(), input_tool=api.input_tool)


# the endpoint stays an ordinary request/response handler
@app.post("/sessions/{id}/research")
async def research(id, body):
    try:
        async with api.session(session_id=id) as session:
            await session.accept_input(body.input)
            return await service.run(body.task)    # resumes where it paused
    except InputRequired as e:
        return {"status": "needs_input", "prompt": e.prompt}
```

The service has no checkpoint code, step keys, or 202 plumbing. Each LLM call and
tool call is engraved automatically; on re-invocation the completed ones **replay
their exact outputs** (the approved draft is the same draft) and only the unfinished
step runs. Idempotency does not disappear entirely — a side effect that runs but
crashes before it commits still needs a key at that boundary (see the traps above) —
but it stays localized to the side-effecting step instead of spreading across the
turn. The pause is just the input tool raising when no input is recorded yet.

## What sefia removes

| The turn must…                          | Hand-rolled                          | sefia |
| --------------------------------------- | ------------------------------------ | ----- |
| Skip completed steps                    | `if "x" not in r:` per step          | automatic (replay) |
| Replay exact LLM/tool outputs           | persist & cache every result         | automatic (content-addressed) |
| Fire side effects once                  | per-step idempotency keys            | replay of completed steps; a key only at the side-effect boundary |
| Survive a process restart               | load state, reconstruct the loop     | re-invoke; completed work replays |
| Pause / resume for a human              | `Paused`, `approval=None`, 202 dance | a tool raises; re-invoke resumes |
| Not double-run on concurrent re-entry   | a lock you write                     | session-scoped |

Most of that resume logic exists for *correctness* (non-determinism, exactly-once),
not convenience. Your database still stores your data; sefia stores just enough to
bring a paused or crashed turn back to where it was.
