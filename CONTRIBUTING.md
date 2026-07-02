# Contributing

Thanks for looking at sefia. This is the human-facing development guide; the layout
reference is [`docs/architecture.md`](./docs/architecture.md) and the runtime is
explained in [`docs/how-it-works.md`](./docs/how-it-works.md).

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (the repo is a `uv` workspace)

## Setup & commands

```bash
uv sync                       # install the workspace
uv run pytest                 # run all tests (asyncio auto-mode)
uv run pytest packages/sefia  # run one package's tests
uv run ruff check .           # lint
uv run pyright                # type-check
```

Tests mirror the source under each package's `tests/units/` (per-module) and
`tests/scenarios/` (behavioral). Add tests next to the layer you change.

## Where to make a change

The per-module map and the **where-to-change-what** table are in
[`docs/architecture.md`](./docs/architecture.md). Start there to find the right file,
then read [`docs/how-it-works.md`](./docs/how-it-works.md) if you need the runtime
data flow.

## Conventions & guardrails

The full list (dependency direction, the two seams, internal vs. public naming) lives
in [`docs/architecture.md`](./docs/architecture.md#conventions). The load-bearing ones:

- **Keep dependency arrows one-way.** `sefia` core must not import `sefios`,
  `sefia_litellm`, or any provider SDK. Provider/IO/convenience concerns go in an
  adapter or in `sefios`.
- **Provider specifics stay behind interfaces** — anything LiteLLM/OpenAI-shaped lives
  in `sefia_litellm` behind `LLMClient`.
- **Respect the two seams** — *middleware* controls (may retry/short-circuit),
  *handlers* only observe. Don't route control through a handler.
- **Underscore = internal.** Import public names from a package's `__init__.py`.
- Match the surrounding code's style and density; don't introduce a workflow engine or
  break the stateless-HTTP model.

## Keep the docs in sync

Several docs describe the code at a level that **drifts when the code changes** —
treat updating them as part of the change, not a follow-up. The mapping:

| If you change… | Update… |
| --- | --- |
| The public API, exports, or a quickstart-level usage | `README.md`, `docs/tutorial.md` |
| The runtime mechanism (executor, strategy, decorators, context, glyff glue) | `docs/how-it-works.md` (it references specific modules/behavior) |
| Package layout, a module's role, or the dependency graph | `docs/architecture.md` (and `CONTRIBUTING.md`/`AGENTS.md` if commands change) |
| The tool-exposure model | `DESIGN.md`, `README.md`, the tool sections of `docs/how-it-works.md`, and the relevant issue |
| A tradeoff or positioning claim | `DESIGN.md` (non-goals), `docs/tradeoffs.md`, `docs/choosing.md`, `docs/faq.md` |

When in doubt, grep the docs for the symbol or filename you touched. A change that
makes a doc's example or file reference wrong is incomplete until the doc is fixed.

## Pre-1.0

The API is unstable and parts of the design — notably the tool-exposure rule — are
being finalized. Check [`DESIGN.md`](./DESIGN.md) and the open issues before changing
tool discovery or other in-flight areas.

## Pull requests

Keep PRs focused, include tests for behavior changes, and make sure
`pytest` / `ruff` / `pyright` pass. Describe the change and the reasoning; link any
relevant issue.
