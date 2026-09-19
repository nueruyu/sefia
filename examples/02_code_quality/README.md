# Sefia Example: Code Quality Review

This example builds a small LLM workflow with `sefia` for reviewing code
quality.

It clarifies the review scope, reads tracked project files, runs multiple review
perspectives, and produces a final quality report.

## Commands

Run these commands from the repository root. See the [examples README](../README.md)
for setup.

### Start or Resume Chat

```bash
uv run python -m examples.02_code_quality.main chat
```

If the workflow asks for human input, use the reported interaction ID with your result:

```bash
uv run python -m examples.02_code_quality.main chat "Use E:/path/to/project and focus on the Python files" --interaction-id "<interaction_id>"
```

### Verbose Mode

```bash
uv run python -m examples.02_code_quality.main chat --verbose
```

### Sessions

```bash
uv run python -m examples.02_code_quality.main session new
uv run python -m examples.02_code_quality.main session switch <session-id>
```

Start without a message to create a durable request. Supply text only with the
reported `--interaction-id`; pre-request input and implicit routing are unsupported.
