# Human-in-the-loop without re-running the turn

*The hard part of agents isn't calling the model — it's resuming a turn that
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

- **Every LLM/tool output must be persisted and replayed** — for *correctness*,
  not just cost. Forget to cache `draft` and a resume regenerates it; the human
  approved a draft that no longer exists.
- **Persist after every step.** A crash between "do work" and "save" re-runs that
  step. Fine for pure reads; **every side-effecting step then needs its own
  idempotency key** (step 5).
- **Step keys are load-bearing.** Reorder or insert a step and the saved progress
  silently maps to the wrong slot.
- **Concurrent re-entry double-runs.** Two requests resuming the same session need
  a lock you haven't written yet.
- **The pause plumbing leaks into the signature** (`approval=None`, `Paused`, the
  202 dance) and into every caller.
- **This is one happy path.** Add a second tool, a loop, a retry, or a sub-agent,
  and the `if "x" not in r` bookkeeping multiplies — you are now maintaining a
  small, bespoke durable-execution engine.

### The framework alternatives

A durable workflow engine (or a typed-agent framework's durable wrapper) does all
of the above correctly — at the cost of adopting that engine and its
infrastructure. A graph framework gives you a built-in checkpointer and an
interrupt — at the cost of authoring the flow as a graph. Both are reasonable; see
[02 — approval-gated workflow](./02-approval-gated-workflow.md) for how they read.

## With sefia

The turn is an ordinary typed function; the pause is a tool that raises:

```python
class Research:
    def __init__(self, web: WebToolkit, human: HumanInput):
        self._web = web
        self._human = human

    @infer
    async def run(self, task: str) -> Report:
        """Clarify the task, research it with web search, draft a report, get the
        human's approval of the draft, then finalize and send."""
        ...

# the endpoint stays an ordinary request/response handler
@app.post("/sessions/{id}/research")
async def research(id, body):
    async with api.session(session_id=id) as s:
        await s.accept_input(body.approval)        # answer for a pending question
        return await agent.run(body.task)          # resumes where it paused
```

There is no checkpoint code, no step keys, no idempotency keys, and no 202
plumbing in the agent. Each LLM call and tool call is engraved automatically; on
re-invocation the completed ones **replay their exact outputs** (the approved
draft is the same draft) and only the unfinished step runs. The pause is just the
human-input tool raising when no answer is recorded yet.

## What collapsed

| The turn must…                          | Hand-rolled                          | sefia |
| --------------------------------------- | ------------------------------------ | ----- |
| Skip completed steps                    | `if "x" not in r:` per step          | automatic (replay) |
| Replay exact LLM/tool outputs           | persist & cache every result         | automatic (content-addressed) |
| Fire side effects once                  | per-step idempotency keys            | replay of the completed step |
| Survive a process restart               | load state, reconstruct the loop     | re-invoke; completed work replays |
| Pause / resume for a human              | `Paused`, `approval=None`, 202 dance | a tool raises; re-invoke resumes |
| Not double-run on concurrent re-entry   | a lock you write                     | session-scoped |

## The point

The resume logic is the hard, error-prone part — and most of it exists for
*correctness* (non-determinism, exactly-once), not convenience. sefia removes it:
you write the turn as a normal typed function, pausing is just raising, and
resuming is automatic. Your database still stores your data; sefia stores just
enough to bring a paused or crashed turn back to exactly where it was.
