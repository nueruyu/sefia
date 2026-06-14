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
python -m examples.02_code_quality.main chat "Review this repository for maintainability issues"
```

If the workflow asks for human input, run the same command with your answer:

```bash
python -m examples.02_code_quality.main chat "Use E:/path/to/project and focus on the Python files"
```

### Verbose Mode

```bash
python -m examples.02_code_quality.main chat "Review this repository for maintainability issues" --verbose
```

### Sessions

```bash
python -m examples.02_code_quality.main session new
python -m examples.02_code_quality.main session switch <session-id>
```
