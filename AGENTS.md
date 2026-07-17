# AGENTS.md

AI coding agents follow the same guidance as human contributors:

- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — setup, commands, conventions, and guardrails.
- [`docs/architecture.md`](./docs/architecture.md) — package layout and where to change
  what. It links the rest ([docs index](./docs/README.md), how it works, design).

## Comments & docstrings

Keep them concise and intentional. A comment earns its place by explaining
something the code cannot.

- Don't restate what the code, an assertion, or a test already makes clear.
- Don't repeat rationale that's already captured in an issue or PR — link it instead.
- Do comment non-obvious constraints, invariants, trade-offs, or behavior that naming
  and structure can't convey.
- The same rule applies to test comments and docstrings.

## Language

Write commit messages and pull request titles and descriptions in English.
