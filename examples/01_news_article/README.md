# Sefia Example: News Article Generation

This example shows how to build a small LLM workflow with `sefia` using ordinary
async Python calls. It includes request clarification, real web search, writing,
and session interruption/resumption for human input.

## Scenario

The goal is to create a news article from a clarified user request. The process
is a normal call graph with three LLM-backed steps:

1. `RequirementsClarifier.clarify_request` organizes the user's initial request
   and asks focused follow-up questions with a `HumanInputTool` until critical
   ambiguities are resolved.
2. `Researcher.research_topic` uses a `WebSearchTool` to find relevant sources online.
3. `NewsWriter.write_article` drafts an article from the clarified request and
   sources, then can ask the user for feedback and resume after input is
   provided.

## How to Run

### 1. Initial Setup

Ensure you have set up the development environment from the root of the repository. This will install `sefia`, `sefia_litellm`, `glyff`, and other dependencies.

```bash
# Make sure you are in the root directory of the sefia monorepo
uv venv
uv pip install -e "packages/sefia" -e "packages/sefia_litellm"
uv pip install -e "examples"
```

You also need an LLM API key. Set it as an environment variable (e.g., for OpenAI):

```bash
export OPENAI_API_KEY="your-api-key"

# Optional: Set a default model for the examples
export EXAMPLE_DEFAULT_MODEL="gpt-4o-mini"
```

### 2. Start a New Workflow

To start a new workflow, use the `chat` command and provide a topic. If no session is active, a new one will be created automatically.

```bash
python examples/01_news_article/main.py chat "The impact of generative AI on software development"
```

The workflow first clarifies your request. If your initial topic is underspecified, it asks focused follow-up questions before research begins. If it needs your input, it pauses and saves the active session ID automatically.

**Example Output (Clarification):**

```text
> No active session. Starting new session: a1b2c3d4-...
> Stage 1: Clarifying request...

[USER_INPUT_REQUIRED] Who is the target audience for this article?

╭──────────────────────────── WAITING FOR INPUT ────────────────────────────╮
│ Session interrupted to wait for your input.                               │
│ To resume, run the script again with your answer.                         │
╰───────────────────────────────────────────────────────────────────────────╯
```

Resume with your answer. The clarifier can ask additional questions until the request is clear enough to proceed.

```bash
python examples/01_news_article/main.py chat "Software engineering managers"
```

When the request is clear, research and writing continue:

```text
> Resuming session a1b2c3d4-...
> Stage 1: Clarifying request...
   -> Clarified request:

Topic: The impact of generative AI on software development
Angle: Practical impact on productivity, testing, and engineering workflows
Audience: Software engineering managers
Requirements:
- Include potential risks such as code quality and security vulnerabilities.

> Stage 2: Researching topic...
   -> Found sources: ['https://...', 'https://...']
> Stage 3: Writing article...
```

### 3. (Optional) Enable Verbose Debugging

To see the exact prompts being sent to the LLM for debugging purposes, add the `--verbose` flag to the `chat` command.

```bash
python examples/01_news_article/main.py chat "The impact of generative AI on software development" --verbose
```

**Example Output (Verbose Prompt Dump):**

```text
┌──────────────────── LLM PROMPT ────────────────────┐
│ [SYSTEM]                                           │
│ Research the clarified article request to find     │
│ relevant online sources.                           │
│ ...                                                │
│                                                    │
│ [USER]                                             │
│ Task arguments:                                    │
│ - article_request: Topic: The impact of ...        │
└────────────────────────────────────────────────────┘
```

### 4. Resume the Workflow

To resume, use the `chat` command again with your answer. The workflow automatically picks up the active session.

```bash
python examples/01_news_article/main.py chat "Please add a section about the potential risks, like code quality and security vulnerabilities."
```

The workflow incorporates your feedback and runs until it either finishes or requires more input.

**Example Output (Completion):**

```text
> Resuming session a1b2c3d4-...
> Stage 3: Writing article...
{"final_answer": {"title": "The Transformative Impact of Generative AI on Software Development", "summary": "...", "sources": ["..."]}}

--- FINAL ARTICLE ---
Title: The Transformative Impact of Generative AI on Software Development
Summary: Generative AI is revolutionizing software development...
Sources: https://..., https://...
---
```

### 5. Managing Sessions

You can create a fresh active session or switch between existing sessions if you have started multiple topics.

```bash
# Create and switch to a new session
python examples/01_news_article/main.py session new

# Switch to a different, existing session
python examples/01_news_article/main.py session switch <another-session-id>
```
