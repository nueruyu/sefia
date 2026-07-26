# An approval-gated workflow that survives a restart

*draft (with web search) → critique → revise → human approval → publish*, where the
process may restart before the human approves.

## What it has to do

- run several model/tool steps to produce a draft;
- pause for a human to approve;
- resume after a restart (a deploy, a crash, or the approval arriving in a new request);
- not re-run the completed steps — the human approved *that* draft, and re-running a
  non-deterministic step would produce a different one.

## Hand-rolled, this usually means

- per-step cached outputs;
- idempotency keys at side-effect boundaries;
- persisted resume state;
- a lock against concurrent re-entry.

## With sefia

- model and tool calls replay from the store, so completed steps don't re-run;
- the pause is a tool raising `InputRequired`;
- resume is re-invoking the same session;
- the orchestration stays ordinary Python.

```python
async def research(task, drafter, human) -> Outcome:
    d = await drafter.draft(task)
    c = await critique(d)
    if c.needs_work:
        d = await revise(d, c)
    if await human.ask(f"Publish?\n\n{d.text}") == "yes":   # pause, then resume
        return await publish(d)
    return Outcome(published=False)
```

For a graph-first version, LangGraph may fit better; for a typed agent runtime with a
durable backend, Pydantic AI + Temporal or DBOS. See [choosing.md](../choosing.md).
