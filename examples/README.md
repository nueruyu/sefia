# Sefia Examples

These examples demonstrate small LLM workflows built with `sefia`.

Run commands from the repository root. Install dependencies with the root
project setup, such as `uv sync`.

## Setup

Set an LLM API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Optionally set the model used by examples:

```bash
export EXAMPLE_DEFAULT_MODEL="gpt-4o-mini"
```

## Examples

- [00 Simple Chat](./00_simple_chat/README.md)
- [01 News Article Generation](./01_news_article/README.md)
- [02 Code Quality Review](./02_code_quality/README.md)
- [03 FastAPI API](./03_fastapi_api/README.md)

## Tests

The example tools and the rendering helpers are covered by unit tests, and each
example's workflow is covered by a scenario test that mocks the agents' `@infer`
methods. (The CLI and HTTP plumbing the examples build on lives in `sefios.cli`
and `sefios.fastapi`; it is tested alongside those packages.) Everything runs
without any LLM API calls. Run them from the repository root:

```bash
uv run pytest examples/tests
```
