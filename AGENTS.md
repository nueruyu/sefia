# AGENTS.md

Orientation for AI coding agents (and new contributors) working in this repo. Read
this first, then [`docs/architecture.md`](./docs/architecture.md) for the full map.

## What this is

sefia turns typed async Python functions into LLM-backed calls (`@infer`) that are
durable and resumable — pause for a human, survive a restart — over plain stateless
HTTP, with no workflow engine. A `uv` workspace of small packages:
**sefia** (core) → **sefia_litellm** (provider adapter); **sefia** → **sefios**
(batteries); **jsonstream** (standalone). Durability comes from **glyff** (separate
repo, a dependency). See [`README.md`](./README.md) and [`DESIGN.md`](./DESIGN.md).

## Commands

```bash
uv sync                       # install the workspace
uv run pytest                 # all tests (asyncio auto-mode)
uv run pytest packages/sefia  # one package
uv run ruff check .           # lint
uv run pyright                # type-check (pyright config in pyproject.toml)
```

Tests mirror source under each package's `tests/units/` (per-module) and
`tests/scenarios/` (behavioral). Add tests beside the layer you change.

## Where to make a change

Use the **"if you want to change X, look at Y"** table and the per-module map in
[`docs/architecture.md`](./docs/architecture.md). For the runtime data flow (the
`@infer` loop, the unified schema, engrave/replay, the pause mechanism) see
[`docs/how-it-works.md`](./docs/how-it-works.md).

## Guardrails

- **Keep dependency arrows one-way.** `sefia` core must not import `sefios`,
  `sefia_litellm`, or any provider SDK. Provider/IO/convenience concerns go in an
  adapter or in `sefios`.
- **Provider specifics stay behind interfaces.** Anything LiteLLM/OpenAI-shaped lives
  in `sefia_litellm` behind `LLMClient`; the core only knows the interface.
- **Respect the two seams.** *Middleware* controls (may retry/short-circuit);
  *handlers* only observe (exceptions isolated). Don't route control through a handler.
- **Underscore = internal.** Import public names from a package `__init__.py`; don't
  reach into `_module` internals across package boundaries.
- **Pre-1.0.** The tool-exposure rule (`@tool` marker → public methods of held
  objects) is still being finalized; check [`DESIGN.md`](./DESIGN.md) and open issues
  before changing tool discovery.

## Scope of changes

Match the surrounding code's style and density. Don't add a provider dependency to the
core, don't introduce a workflow engine, and don't break the stateless-HTTP model
(nothing should need to run in the background between requests).
