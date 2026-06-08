# Sefia Example: News Article Generation

This example builds a small LLM workflow with `sefia` using ordinary async
Python calls.

It clarifies a news article request, searches the web for sources, writes the
article, and can pause/resume when human input is needed.

## Commands

Run these commands from the repository root. See the [examples README](../README.md)
for common setup.

### Start or Resume Chat

```bash
python -m examples.01_news_article.main chat "The impact of generative AI on software development"
```

If the workflow asks for human input, run the same command with your answer:

```bash
python -m examples.01_news_article.main chat "Software engineering managers"
```

### Verbose Mode

```bash
python -m examples.01_news_article.main chat "The impact of generative AI on software development" --verbose
```

### Sessions

```bash
python -m examples.01_news_article.main session new
python -m examples.01_news_article.main session switch <session-id>
```
