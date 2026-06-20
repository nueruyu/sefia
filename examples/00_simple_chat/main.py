"""
00_simple_chat — simplest Sefia example.

Each CLI invocation sends one message and the session pauses waiting for the
next. Run it again (with --reply-to) to continue the conversation.
"""

import asyncio
from pathlib import Path

import typer
from typing_extensions import Annotated

from .._common.sefia_cli import SefiaCLI
from .._common.session import UnknownSessionError
from .agent import ChatAgent

SESSION_DIR = Path(__file__).parent / ".local"
sefia_cli = SefiaCLI(session_dir=SESSION_DIR, stream=True)
human_input_tool = sefia_cli.human_input_tool

agent = ChatAgent(human_input_tool)

app = typer.Typer(help="Simple one-agent chat loop.")


@app.command()
def chat(
    message: Annotated[
        list[str],
        typer.Argument(help="Your message, or an answer to resume the session."),
    ],
    reply_to: Annotated[
        str | None,
        typer.Option("--reply-to", help="Human input interaction ID to answer."),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option("--session-id", help="Session to use (default: active session)."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            envvar="EXAMPLE_DEFAULT_MODEL",
            help="LLM model to use.",
        ),
    ] = "gpt-4o-mini",
):
    """Send a message to the chat agent."""
    asyncio.run(
        _chat_async(
            message=message,
            reply_to=reply_to,
            session_id=session_id,
            model=model,
        )
    )


async def _chat_async(
    *,
    message: list[str],
    reply_to: str | None,
    session_id: str | None,
    model: str,
) -> None:
    async with sefia_cli.session(session_id=session_id, model=model) as session:
        await session.accept_input(message, reply_to=reply_to)
        await agent.chat()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
