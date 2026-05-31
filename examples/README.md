# Sefia Workflow Sample

This example demonstrates a multi-agent workflow using `sefia` that includes real web searches, agent collaboration, and session interruption/resumption for human-in-the-loop interaction.

## Scenario

The goal is to create a news article on a given topic. The process involves two agents:

1. A **Researcher** agent that uses a `WebSearchTool` to find relevant sources online.
2. A **Writer** agent that drafts an article based on the research and uses a `HumanInputTool` to ask the user for feedback, interrupting the session until the user provides input.

The entire workflow is orchestrated in a simple `main` function, showcasing how to chain agent calls.

## How to Run

### 1. Initial Setup

Ensure you have set up the development environment from the root of the repository. This will install `sefia`, `glyff`, and other dependencies like `duckduckgo-search`.

```bash
# Make sure you are in the root directory of the sefia monorepo
uv venv
uv pip install -e .
```

You also need an LLM API key. Set it as an environment variable (e.g., for OpenAI):

```bash
export OPENAI_API_KEY="your-api-key"
```

### 2. Start the Workflow

Run the script with a topic.

```bash
python examples/main.py --topic "The impact of generative AI on software development"
```

The script will first execute the **Researcher** agent, which performs a live web search. Then, it will pass the results to the **Writer** agent. The workflow will pause when it requires your feedback, print a message, and exit, showing the session ID.

### 3. (Optional) Enable Streaming

To see the LLM's thought process and final answer streamed token by token, add the `--stream` flag. This is useful for observing the agent's behavior in real-time.

```bash
python examples/main.py --topic "The impact of generative AI on software development" --stream
```

**Example Output (Interruption):**

```
> Stage 1: Researching topic...
   -> Found sources: ['https://...', 'https://...']
> Stage 2: Writing article...
{"tool_calls": [{"name": "HumanInputTool_ask_user", "arguments": {"question": "I have drafted an article on how generative AI is accelerating coding and testing. What specific aspects should I focus on for the final version?"}}]}

[USER_INPUT_REQUIRED] I have drafted an article on how generative AI is accelerating coding and testing. What specific aspects should I focus on for the final version?

---
Session interrupted to wait for your input.
To resume, run the script again with the session ID and your answer:
python examples/main.py --session-id "a1b2c3d4-..." --answer "Your answer"
---
```

### 4. Resume the Workflow

Use the provided `session-id` and add your answer to resume the process. You can also use the `--stream` flag here.

```bash
python examples/main.py --session-id "a1b2c3d4-..." --answer "Please add a section about the potential risks, like code quality and security vulnerabilities." --stream
```

The workflow will resume from where it left off, incorporating your feedback into the final article, which will then be printed to the console.

**Example Output (Completion):**

```
> Resuming session a1b2c3d4-...
> Stage 2: Writing article...
{"final_answer": {"title": "The Transformative Impact of Generative AI on Software Development", "summary": "...", "sources": ["..."]}}

--- FINAL ARTICLE ---
Title: The Transformative Impact of Generative AI on Software Development
Summary: Generative AI is revolutionizing software development by accelerating coding and testing phases. This article explores these benefits while also addressing the potential risks, including challenges in maintaining code quality and ensuring security against new vulnerabilities introduced by AI-generated code.
Sources: https://..., https://...
---
```
