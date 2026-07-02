# Use cases

Where sefia's tradeoff (*automatic resume for LLM-backed calls, no engine, no graph*) helps,
shown against hand-rolled code and other approaches, and clear about where it doesn't
fit.

- [01 — Human-in-the-loop without re-running the turn](./01-human-in-the-loop.md)
  — why resumption is the hard part, hand-rolled vs sefia.
- [02 — An approval-gated workflow that survives a restart](./02-approval-gated-workflow.md)
  — the workflow shape, where sefia fits, and where graph, agent, or workflow layers
  may fit better.

> Non-sefia descriptions are category-level. APIs move fast — check each tool's docs
> for specifics.
