# An approval-gated workflow that survives a restart

A multi-step LLM pipeline that **pauses for human approval** and must **resume
after a process restart** (a deploy, a crash, or the approval arriving in a new
request):

*draft (with web search) → critique → revise → human approval → publish.*

> The non-sefia snippets sketch **LangGraph** and **Pydantic AI** (with Temporal /
> DBOS for durability). Their APIs are approximate and move fast — faithful in shape,
> not copy-paste; check each tool's docs for specifics.

## sefia

Tools are scoped to the object that holds them, so only the draft step holds
`web`. `critique` / `revise` need no tools (plain `@infer` functions). The human
pause is a deterministic call in the orchestrator. Resume is automatic.

```python
class Drafter:
    def __init__(self, web: WebSearch):
        self._web = web                 # tool — only the draft step sees it

    @infer
    async def draft(self, task: str) -> Draft:
        """Research the task with web search, then draft a report."""
        ...

@infer
async def critique(draft: Draft) -> Critique:
    """Critique the draft for gaps and errors."""
    ...

@infer
async def revise(draft: Draft, critique: Critique) -> Draft:
    """Revise the draft to address the critique."""
    ...

async def research(task, drafter: Drafter, human: HumanInput) -> Outcome:
    d = await drafter.draft(task)
    c = await critique(d)
    if c.needs_work:
        d = await revise(d, c)
    if await human.ask(f"Publish?\n\n{d.text}") == "yes":   # pause — durable
        return await publish(d)
    return Outcome(published=False)
```

```python
@app.post("/sessions/{id}/research")
async def endpoint(id, body):
    async with scope.session(session_id=id) as s:
        await s.accept_input(body.approval)
        return await research(body.task, drafter, human)   # resumes; steps replay
```

Adding the human approval was one line. Adding "survive a restart" was nothing —
the run is engraved, so a paused run resumes in a fresh process and the completed
steps replay.

## LangGraph

A shared `State`, a node per step (with bodies), the edges, and LangGraph's
checkpointer + interrupt for the durable pause.

```python
class State(TypedDict):
    task: str
    draft: Draft | None
    critique: Critique | None
    approved: bool | None

async def draft_node(s: State) -> State:
    s["draft"] = await research_and_draft(s["task"])   # calls web_search
    return s

async def critique_node(s: State) -> State:
    s["critique"] = await llm_critique(s["draft"])
    return s

async def revise_node(s: State) -> State:
    s["draft"] = await llm_revise(s["draft"], s["critique"])
    return s

async def approve_node(s: State) -> State:
    s["approved"] = interrupt(f"Publish?\n\n{s['draft'].text}")  # pause
    return s
```

```python
g = StateGraph(State)
g.add_node("draft", draft_node)
g.add_node("critique", critique_node)
g.add_node("revise", revise_node)
g.add_node("approve", approve_node)

g.add_edge(START, "draft")
g.add_edge("draft", "critique")
g.add_conditional_edges(
    "critique",
    lambda s: "revise" if s["critique"].needs_work else "approve",
)
g.add_edge("revise", "approve")
g.add_conditional_edges(
    "approve",
    lambda s: "publish" if s["approved"] else END,
)
app = g.compile(checkpointer=SqliteSaver(db))       # persistence backend

await app.invoke({"task": task}, config={"thread_id": id})
await app.invoke(Command(resume=approval), config={"thread_id": id})
```

Durable interrupt/resume is a real strength — the cost is authoring the graph,
threading `State`, and operating the runtime's persistence.

## Pydantic AI

Each step is a configured agent; the deterministic flow is plain orchestration.

```python
drafter = Agent(
    model="…",
    output_type=Draft,
    tools=[web_search],
    instructions="Research the task, then draft a report.",
)
critic = Agent(
    model="…",
    output_type=Critique,
    instructions="Critique the draft for gaps and errors.",
)
reviser = Agent(
    model="…",
    output_type=Draft,
    instructions="Revise the draft per the critique.",
)
```

To make it **durable** Pydantic AI wraps each agent in an engine — a one-line
`DBOSAgent` / `TemporalAgent` wrapper, plus a durable workflow for the orchestration
and the human wait:

```python
drafter = DBOSAgent(drafter)        # or TemporalAgent — wraps run(); model/tool
critic  = DBOSAgent(critic)         # calls become checkpointed steps
reviser = DBOSAgent(reviser)

@DBOS.workflow                       # Temporal: @workflow.defn
async def research(task: str) -> Outcome:
    draft = (await drafter.run(task)).output
    critique = (await critic.run(draft)).output
    if critique.needs_work:
        draft = (await reviser.run((draft, critique))).output
    if (await DBOS.recv("approval")) == "yes":   # durable wait — Temporal: wait_condition
        return await publish(draft)
    return Outcome(published=False)
```

With first-class durable wrappers the **orchestration code converges** toward
sefia's: plain async with durable awaits, no hand-rolled resume engine. What diverges
is **operations**: you adopt a durable-execution engine and run its
infrastructure. With **DBOS** that is a library plus a Postgres database; with
**Temporal** it is a server cluster plus workers. (Pydantic AI also has a native
*deferred tools* path that stays stateless-HTTP with no engine, at the cost of
threading the message history yourself — see the [FAQ](../faq.md). Without either,
you hand-roll the resume engine — see [01](./01-human-in-the-loop.md).)

## What the restart requirement cost each

| | the workflow code | how it becomes durable | what you operate |
| --- | --- | --- | --- |
| **sefia** | functions + plain `await` | native (engraved) | an HTTP handler + a store (sqlite/file/PG) |
| **LangGraph** | nodes + edges + `State` | built-in checkpointer + interrupt | the graph runtime + its store |
| **Pydantic AI + DBOS** | N agents + a durable workflow | `DBOSAgent(agent)` wrapper | the DBOS library + **Postgres** |
| **Pydantic AI + Temporal** | N agents + a durable workflow | `TemporalAgent(agent)` wrapper | a **Temporal cluster + workers** |

## Where each is the right call

- **LangGraph** — when you want the flow as an operable artifact *and* durable
  interrupt from the framework. Strong at pause/resume; you adopt and
  operate the graph runtime.
- **Pydantic AI** — when the flow is **model-driven** (the model picks tools and
  loops). Durability is a clean wrapper away, and for an already-Postgres or
  already-Temporal team that path is very reasonable.
- **sefia** — when the workflow is a typed Python call graph you want durable and
  resumable on a plain stateless handler, with **no engine to adopt** and no
  Postgres or cluster required. You give up the graph's operable diagram and the
  agent runtime's open-ended loop.

## Summary

The dividing line is **how you get durability**, and it is *not* code length —
with first-class wrappers the Pydantic AI code is about as short as sefia's. It
is what you adopt and operate: sefia's durability is **native to the model** and
runs on a plain handler with a sqlite/file store; the Pydantic AI path adopts a
durable-execution engine (DBOS + Postgres, or a Temporal cluster); LangGraph adopts
its own stateful runtime. sefia is clearly lighter than Temporal; against DBOS the
gap is small (DBOS is itself light) and comes down to "no engine, no Postgres
requirement, stateless-HTTP-native" plus the typed-function ergonomics. Pick
LangGraph when the flow *is* the artifact; Pydantic AI when the model drives;
sefia when you want a durable workflow that stays ordinary Python with nothing to
operate but a store.
