# 00 Simple Chat

The smallest Sefia chat example: a terminal conversation loop backed by one
`@infer` agent and the shared CLI session helpers.

Each `chat` command sends one message into the active session. The agent replies,
then pauses until you run the command again with the next message.

## Run

Run from the repository root. See the [examples README](../README.md) for setup.

```bash
python -m examples.00_simple_chat.main chat "Hello"
```

You can choose a model with either `--model` or `EXAMPLE_DEFAULT_MODEL`:

```bash
EXAMPLE_DEFAULT_MODEL=gpt-4o-mini python -m examples.00_simple_chat.main chat "Hello"
```

## Sessions

The example uses `SefiaCLI`, so conversation state is persisted under the
example's local session directory and can be resumed by later invocations.

```bash
python -m examples.00_simple_chat.main session new
python -m examples.00_simple_chat.main session switch <session-id>
```

## What it shows

- Defining an agent with `@infer`
- Running a persisted CLI session with `SefiaCLI`
- Feeding terminal input through `HumanInputTool`
- A minimal single-agent loop without external domain tools or multi-agent orchestration
