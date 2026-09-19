# Sefia Example: News Article Generation

This example builds a small LLM workflow with `sefia` using ordinary async
Python calls.

It clarifies a news article request, searches the web for sources, writes the
article, and can pause/resume when human input is needed.

## Commands

Run these commands from the repository root. See the [examples README](../README.md)
for setup.

### Start or Resume Chat

```bash
uv run python -m examples.01_news_article.main chat
```

If the workflow asks for human input, use the reported interaction ID with your result:

```bash
uv run python -m examples.01_news_article.main chat "Software engineering managers" --interaction-id "<interaction_id>"
```

### Verbose Mode

```bash
uv run python -m examples.01_news_article.main chat --verbose
```

### Sessions

```bash
uv run python -m examples.01_news_article.main session new
uv run python -m examples.01_news_article.main session switch <session-id>
```

Start without a message to create a durable request. Supply text only with the
reported `--interaction-id`; pre-request input and implicit routing are unsupported.
