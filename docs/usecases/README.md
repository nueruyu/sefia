# Use cases

Where sefia's tradeoff (*automatic resume for agents, no engine, no graph*) helps,
shown against hand-rolled code and other approaches, and clear about where it doesn't
fit.

- [01 — Human-in-the-loop without re-running the turn](./01-human-in-the-loop.md)
  — why resumption is the hard part, hand-rolled vs sefia.
- [02 — An approval-gated workflow that survives a restart](./02-approval-gated-workflow.md)
  — the same workflow across Pydantic AI, LangGraph, and sefia.

See also [the positioning argument](../why-less.md) for the
positioning argument behind these examples: concept surface, provider leakage, and
operational weight.

> Non-sefia snippets sketch LangGraph and Pydantic AI (with Temporal / DBOS for
> durability). APIs are approximate and move fast — faithful in shape, not
> copy-paste; check each tool's docs for specifics.
