# Documentation

The [project README](../README.md) is the pitch; this page is the map of everything
else, grouped by what you're trying to do. A reasonable first path:
**[Quickstart](./quickstart.md) → [Design & Philosophy](../DESIGN.md) →
[How it works](./how-it-works.md)**.

## Get started
- **[Quickstart](./quickstart.md)** — from one `@infer` function to a durable
  human-in-the-loop agent over HTTP, step by step.
- **[Examples](../examples/)** — runnable end-to-end agents.

## Understand the model
- **[Design & Philosophy](../DESIGN.md)** — the thesis (`@infer` = an
  LLM-implemented abstract method) and the tool / durability model.
- **[How it works](./how-it-works.md)** — the runtime mechanism, with source
  references: the loop, the unified schema, content-addressed replay.
- **[Architecture map](./architecture.md)** — package layout, dependency direction,
  and where to change what.

## Compare & decide
- **[Concept surface, provider leakage, operational weight](./why-less.md)** — the
  canonical positioning argument, and where an engine is genuinely the right call.
- **[Choosing a stack](./choosing.md)** — when to use sefia vs LangGraph,
  Pydantic AI, DBOS, or Temporal.
- **[FAQ](./faq.md)** — objections and "how does it actually work".
- **[Use cases](./usecases/)** — the same workflows hand-rolled and across
  LangGraph / Pydantic AI / sefia.
- **[Statelessness — a design note](./notes/statelessness.md)** — the vendor-neutral
  tradeoff underneath it all.

## Contribute
- **[Contributing](../CONTRIBUTING.md)** — setup, commands, conventions, and the
  "keep the docs in sync" rule.
- **[Architecture map](./architecture.md)** — where things live.

> Pre-1.0: docs show the **release-target API**. Some surfaces still differ today —
> see each doc's status note and [DESIGN.md](../DESIGN.md).
