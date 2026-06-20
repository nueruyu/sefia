# 00 Simple Chat

The simplest Sefia example: a terminal chat loop backed by a single `@infer` agent.

## Run

```bash
EXAMPLE_DEFAULT_MODEL=gpt-4o-mini python -m examples.00_simple_chat.main
```

## What it shows

- Defining an agent with `@infer`
- Running a `SessionScope` session
- A basic input/reply loop — no tools, no multi-agent, no persistence
