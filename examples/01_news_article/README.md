# Sefia Workflow Sample: News Article Generation

This example demonstrates a multi-agent workflow using `sefia` that includes real web searches, agent collaboration, and session interruption/resumption for human-in-the-loop interaction. The command-line interface is designed to be simple, automatically managing session state between runs.

## Scenario

The goal is to create a news article on a given topic. The process involves two agents:

1. A **Researcher** agent that uses a `WebSearchTool` to find relevant sources online.
2. A **Writer** agent that drafts an article based on the research and uses a `HumanInputTool` to ask the user for feedback, interrupting the session until the user provides input.

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

The script will run, and if it needs your input, it will pause and display a message. The active session ID is now automatically saved.

### 3. (Optional) Enable Verbose Debugging

To see the exact prompts being sent to the LLM for debugging purposes, add the `--verbose` flag to the `chat` command.

```bash
python examples/01_news_article/main.py chat "The impact of generative AI on software development" --verbose
```

**Example Output (Interruption):**

```
> No active session. Starting new session: a1b2c3d4-...
> Stage 1: Researching topic...
┌──────────────────────────────────────────────── LLM PROMPT ────────────────────────────────────────────────┐
│ [SYSTEM]                                                                                                   │
│ Research the given topic to find relevant online sources.                                                  │
│ Your goal is to return a list of high-quality URLs related to the topic.                                   │
│ ...                                                                                                        │
│                                                                                                            │
│ [USER]                                                                                                     │
│ Task arguments:                                                                                            │
│ - topic: The impact of generative AI on software development                                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   -> Found sources: ['https://...', 'https://...']
> Stage 2: Writing article...
{"tool_calls": [{"name": "HumanInputTool_get_human_input", "arguments": {"question": "I have drafted an article on how generative AI is accelerating coding and testing. What specific aspects should I focus on for the final version?"}}]}

[USER_INPUT_REQUIRED] I have drafted an article on how generative AI is accelerating coding and testing. What specific aspects should I focus on for the final version?

╭────────────────────────────────────────── WAITING FOR INPUT ───────────────────────────────────────────╮
│ Session interrupted to wait for your input.                                                            │
│ To resume, run the script again with your answer.                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### 4. Resume the Workflow

To resume, simply use the `chat` command again with your answer. The workflow will automatically pick up the active session.

```bash
python examples/01_news_article/main.py chat "Please add a section about the potential risks, like code quality and security vulnerabilities."
```

The workflow will incorporate your feedback and run until it either finishes or requires more input.

**Example Output (Completion):**

```
> Resuming session a1b2c3d4-...
> Stage 2: Writing article...
{"final_answer": {"title": "The Transformative Impact of Generative AI on Software Development", "summary": "...", "sources": ["..."]}}

--- FINAL ARTICLE ---
Title: The Transformative Impact of Generative AI on Software Development
Summary: Generative AI is revolutionizing software development...
Sources: https://..., https://...
---
```

### 5. Managing Sessions

You can switch between different sessions if you have started multiple topics.

```bash
# Switch to a different, existing session
python examples/01_news_article/main.py session switch <another-session-id>
```
