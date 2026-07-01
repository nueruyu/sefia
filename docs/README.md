# Documentation

The [project README](../README.md) is the pitch and the minimal example; this page is
the map of everything else. A reasonable first path:
**[README](../README.md) → [Tutorial](./tutorial.md) → [How it works](./how-it-works.md)**.

## Get started
- **[Tutorial](./tutorial.md)** — build a resumable human-in-the-loop agent, step by
  step.
- **[Examples](../examples/)** — runnable end-to-end agents.

## Understand the model
- **[Design](../DESIGN.md)** — the thesis (`@infer` = an
  LLM-implemented abstract method) and the tool / durability model.
- **[How it works](./how-it-works.md)** — the runtime mechanism, with source
  references: the loop, the unified schema, content-addressed replay.
- **[Architecture map](./architecture.md)** — package layout, dependency direction,
  and where to change what.

## Compare & decide
- **[Concept surface, provider leakage, operational weight](./tradeoffs.md)** — the
  main positioning note, and where an engine is the right call.
- **[Choosing a stack](./choosing.md)** — when to use sefia vs LangGraph,
  Pydantic AI, DBOS, or Temporal.
- **[FAQ](./faq.md)** — objections and "how does it actually work".
- **[Use cases](./usecases/)** — the same workflows hand-rolled and across
  LangGraph / Pydantic AI / sefia.

## Contribute
- **[Contributing](../CONTRIBUTING.md)** — setup, commands, conventions, and the
  "keep the docs in sync" rule.
- **[Architecture map](./architecture.md)** — where things live.

> Pre-1.0: docs show the **release-target API**. Some surfaces still differ today —
> see each doc's status note and [DESIGN.md](../DESIGN.md).
