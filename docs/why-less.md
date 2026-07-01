# Concept surface, provider leakage, and operational weight

A typed-agent framework is judged on three surfaces that have nothing to do with
how good its prompts are: how many concepts you must hold in your head, how much
of the model provider's world leaks into your code, and how much machinery you
have to run to make a paused run survive. sefia's bet is to keep all three small.

> Idiomatic illustrations of each **paradigm**, not any specific library. Neutral
> infrastructure (Temporal, DBOS) is named only to be accurate about operational
> weight. Where a paradigm's strength is real, it is stated as such.

## 1. The concept surface

A typed-agent stack usually asks you to learn a vocabulary before you write a
useful line: an `Agent` object, a way to register tools, a dependency-injection
context threaded into every tool, an `output_type` mechanism, a run/result type,
and, once you want durability, a second vocabulary on top (workflows, steps,
signals, or graph nodes/edges/state).

sefia's vocabulary is mostly Python's own:

| Concept you'd otherwise learn | sefia |
| --- | --- |
| `Agent` construction / config object | a class with an `@infer` method |
| tool registration (decorator + registry) | a public method on a held object |
| DI context threaded into tools | hold the dependency as a field |
| `output_type` / result wrapper | the method's return annotation |
| durable workflow / step / graph DSL | nothing — `await` is already durable |

The two decorators (`@infer`, and glyff's `@engrave` underneath) are the whole
framework-specific surface. Everything else (visibility, types, fields, `await`,
`raise`) is plain Python.

### No per-run DI context

Frameworks that model a tool as a free function need a side channel to pass
per-run dependencies into it — hence a `RunContext`/`Deps` object that every tool
signature carries and every call site threads through. That context exists
*because* tools are detached functions with no `self` to hold state.

sefia's tools are methods on an object you constructed, so the dependency is just
a field:

```python
# context-passing style: the dep rides in a per-run context object
async def search(ctx: RunContext[Deps], q: str) -> list[str]:
    return await ctx.deps.http.get(q)            # reach through the context

# sefia: the dep is held; the method just uses it
class WebToolkit:
    def __init__(self, http: Http):
        self._http = http
    async def search(self, q: str) -> list[str]:
        return await self._http.get(q)           # ordinary attribute access
```

Wiring dependencies is what a constructor (or your DI container) already does.
Re-expressing it as a per-call context parameter is a concept you only need
because the tool wasn't allowed to be a method. Remove that constraint and the
context disappears.

> Tradeoff: a per-run context *does* buy one thing — a tool can read
> run-scoped data (the current run id, usage, a request deadline) without you
> wiring it. sefia surfaces that through an explicit `get_context()` inside a tool
> when needed, rather than putting it in every signature. If most of your tools
> genuinely need run-scoped data, the gap narrows.

## 2. Provider leakage

The second tax is subtler: provider-specific behavior leaking up into the
abstraction you write against. Native tool-calling is per-provider — schema
dialects, strict vs. best-effort structured output, parallel-call support, the
exact shape of a tool-call message — and frameworks that bind to it tend to
surface those differences as caveats on *your* code ("structured output is strict
on these models, best-effort on those"; "parallel tool calls behave differently
here").

sefia deliberately does **not** use native tool-calling. It asks the model for a
single unified result shape (`final_answer | tool_calls`) and uses strict
structured output only where the provider supports it. The consequence is one
abstraction that reads the same across providers: the return type is whatever your
Python type says, full stop, and you don't carry a matrix of per-provider tool
semantics.

```python
@infer
async def classify(self, ticket: str) -> Triage:   # Triage is a normal type;
    """Triage the ticket."""                        # union, nested, whatever
    ...
```

> Tradeoff: this is a real bet, not a free win. By not using native
> tool-calling you give up **native parallel tool calls** and some
> frontier-model tuning on long, complex agent loops, and you take on prompt
> caching as something to design for rather than get for free (tracked, not
> guaranteed). The win is portability and full return-type expressiveness without
> provider leakage; the cost is the native-path optimizations. If your workload is
> one provider and leans on parallel native tools, that calculus can flip.

## 3. Operational weight

The third surface only appears when a run has to **pause and resume across a
process restart** — a human approval, a deploy mid-run, a crash. This is where a
typed-agent stack reaches for a durability engine, and where the operational
weight diverges sharply between paradigms even when the *code* converges.

- **Hand-rolled:** you write the resume engine yourself — per-step caches,
  persist-after-every-step, idempotency keys, a re-entry lock. Most of it exists
  for *correctness* (an LLM call is non-deterministic; re-running the approved
  draft yields a different draft), not just cost. See
  [01 — human-in-the-loop](./usecases/01-human-in-the-loop.md).
- **Durability engine:** a typed-agent framework's durable wrapper does it
  correctly — at the cost of adopting and running that engine. With **DBOS** that
  is a library plus a **Postgres** database; with **Temporal** it is a **server
  cluster plus workers**. The orchestration code gets short, but you now operate
  infrastructure. See [02 — approval-gated workflow](./usecases/02-approval-gated-workflow.md).
- **sefia:** durability is native to the `@infer` model. Engraved calls are
  content-addressed and replay on re-invocation; any exception is non-terminal, so
  **pausing is just raising**. A paused run resumes in a fresh process on a plain
  stateless HTTP handler with a sqlite/file/Postgres store. There is no engine to
  adopt and no cluster or mandatory Postgres to run.

```python
@app.post("/sessions/{id}/turn")
async def turn(id, body):
    async with scope.session(session_id=id) as s:
        await s.accept_input(body.input)
        return await agent.run(body.task)     # resumes where it paused; no engine
```

### When you genuinely need the engine

This is the actual boundary. Reach for a Temporal-grade engine when your workload has
one of these shapes:

- **Long-horizon waits** — a run that sleeps for days or weeks on a durable timer
  (a 30-day trial follow-up, a "ping me next quarter"), not a request-scoped pause.
- **Cross-service sagas with compensation** — a multi-system transaction where a
  late failure must trigger ordered rollbacks/compensations across services.
- **Distributed fan-out across machines** — one logical workflow whose steps must
  run on different workers/hosts and be coordinated, not a single async call graph.
- **Large-scale exactly-once with audit** — high-volume, strict once semantics with
  a durable, queryable execution history as a compliance artifact.

If your workload is one of those, Temporal-grade infrastructure is *justified* and
sefia is the wrong tool — it is single-flow and request/session-scoped by design.
If it is none of those — which the typical agent turn (research, approve, publish;
clarify, act, answer) is not — then an engine is weight you'd operate for
guarantees you don't use, and "durable on a plain handler with a store" is enough.

### Where the paused run lives — and who shares "stateless over HTTP"

The sharpest way to see the operational difference is to ask: **during the pause,
what is running, and where?**

| | what runs during the pause | where the paused run lives | when the server restarts |
| --- | --- | --- | --- |
| **Temporal** | a suspended workflow on a worker | **outside** your HTTP server (cluster) | workflow is untouched; lives in the cluster |
| **DBOS** | a background workflow in your process | **inside** your process + Postgres | recovery rebuilds PENDING workflows from Postgres |
| **sefia** | **nothing** | **nowhere** — state is only in the store | nothing to lose; the next request replays |

This is the concrete meaning of "stateless over HTTP": in sefia a pause is just the
request ending — a tool raises, the handler returns "needs input" + a session id,
and **no background task, worker, daemon, or live workflow exists** between pause
and resume. Resume is an ordinary new request that re-invokes and replays the
engraved steps.

**Scope of the claim: it is not sefia-exclusive.** "Completes over plain
HTTP, nothing running between requests" is a *property*, and two other approaches
share it: hand-rolling the resume yourself (you write the engine — see
[01](./usecases/01-human-in-the-loop.md)), and a typed-agent framework's **native
deferred-tools** path, where `agent.run()` *returns* with the pending request, you
persist the message history, and a new request resumes by passing the result back.
Both are genuinely stateless-HTTP. What separates them from sefia is *within* that
camp, not the camp itself:

- **vs hand-rolled** — you don't write the resume engine.
- **vs deferred-tools** — the pause is an ordinary `raise` at **any point**, not a
  shape tied to "a tool needs approval/external execution"; and memoization is
  **content-addressed over any engraved call** — including your own side-effecting
  Python — not just the model's message history. Same simple client contract, more
  general server.

The frameworks that are *not* in this camp are exactly the engine ones: DBOS keeps
a background workflow plus Postgres; Temporal keeps it in a cluster. They buy the
self-waking durable timer; the stateless-HTTP camp trades that away.

### One mechanism, applied uniformly

Because nothing runs between requests, every kind of "continue later" is the **same
primitive — an HTTP request that re-invokes and replays**: a human answering, a
transient-error retry, and a scheduled wake-up are not three subsystems (a signal,
a retry policy, a durable timer) but one. The cost is that sefia cannot wake
*itself*: a long-horizon "continue in 3 days" needs an external caller (a cron or
scheduler hitting the endpoint). That is a small, ordinary, debuggable moving part —
the same statelessness that removes the engine also turns the timer into "something
calls your URL," which most stacks already have. The boundary where that stops being
enough is the long-horizon/distributed list above.

## The summary

| Surface | Engine / graph stack | sefia |
| --- | --- | --- |
| **Concepts** | Agent + tools + DI context + output_type + workflow/graph vocab | a class, `@infer`, held fields, return types |
| **Leakage** | per-provider tool-call & structured-output semantics surface | one unified result shape; provider-portable (at a stated cost) |
| **Operations** | durable engine + Postgres / cluster + workers | a stateless handler + a store; no engine |

None of this makes sefia a Temporal replacement, and it is not trying to be — it
is the lighter, single-flow, request-scoped layer *before* you need distributed
workflow infrastructure. The claim is narrow: for the pause-and-resume,
human-in-the-loop agent turn that most apps build, sefia keeps the concept surface,
the provider coupling, and the operational weight all low.
