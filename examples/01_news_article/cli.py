import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import glyff.exceptions
import typer
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import Annotated

from .session_setup import setup_session

StateT = TypeVar("StateT", bound=BaseModel)
WriteArticleCallback = Callable[[StateT], Awaitable[tuple[BaseModel, list[str]]]]


# --- Helper functions for session management ---
def _get_session_dir() -> Path:
    session_dir = Path(__file__).parent / ".local"
    session_dir.mkdir(exist_ok=True)
    return session_dir


def _get_active_session_file() -> Path:
    return _get_session_dir() / "active_session.txt"


def _get_active_session_id() -> str | None:
    session_file = _get_active_session_file()
    if session_file.exists():
        return session_file.read_text().strip()
    return None


def _set_active_session_id(session_id: str) -> None:
    _get_active_session_file().write_text(session_id)


# --- Core workflow execution logic ---
async def _run_workflow(
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    stream: bool,
    state_type: type[StateT],
    write_article_callback: WriteArticleCallback[StateT],
):
    """Encapsulates the main logic for running the sefia workflow."""
    if stream:
        print("> Streaming enabled. LLM response will appear below:")
        print("---")

    try:
        async with setup_session(
            model=model, session_id=session_id, stream=stream
        ) as session:
            state_store = session.get_state_store("state", state_type)
            state = await state_store.get()

            if is_new or state is None:
                state = state_type.model_validate({"topic": input_text, "answer": None})
            else:
                state = state.model_copy(update={"answer": input_text})

            await state_store.save(state)

            article, sources = await write_article_callback(state)
            article_data = article.model_dump()

            if stream:
                print("\n---")

            print("\n--- FINAL ARTICLE ---")
            print(f"Title: {article_data['title']}")
            print(f"Summary: {article_data['summary']}")
            print(f"Sources: {', '.join(sources)}")
            print("---")

    except glyff.exceptions.YieldException:
        print("\n---")
        print("Session interrupted to wait for your input.")
        print("To resume, run the script again with your answer:")
        print('python examples/01_news_article/main.py chat "Your answer here"')
        print("---")


def create_app(
    state_type: type[StateT],
    write_article_callback: WriteArticleCallback[StateT],
) -> typer.Typer:
    """Create a CLI app that routes execution to the injected workflow callback."""
    load_dotenv()
    app = typer.Typer(
        help="A multi-agent workflow for generating news articles with human-in-the-loop."
    )
    session_app = typer.Typer(help="Manage user sessions.")
    app.add_typer(session_app, name="session")

    @app.command("chat")
    def chat(
        message: Annotated[
            list[str],
            typer.Argument(
                help="The topic for a new session, or an answer to resume an existing one."
            ),
        ],
        model: Annotated[
            str,
            typer.Option(
                help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
                envvar="EXAMPLE_DEFAULT_MODEL",
            ),
        ] = "gpt-4o",
        stream: Annotated[
            bool, typer.Option(help="Enable streaming of LLM tokens.")
        ] = False,
    ):
        """
        Start a new topic or provide an answer to continue the current session.
        """
        input_text = " ".join(message)
        if not input_text:
            raise typer.BadParameter("Message cannot be empty.")

        session_id = _get_active_session_id()
        is_new = session_id is None

        if is_new:
            session_id = str(uuid.uuid4())
            print(f"> No active session. Starting new session: {session_id}")
            _set_active_session_id(session_id)
        else:
            print(f"> Resuming session {session_id}")

        asyncio.run(
            _run_workflow(
                session_id=session_id,
                input_text=input_text,
                is_new=is_new,
                model=model,
                stream=stream,
                state_type=state_type,
                write_article_callback=write_article_callback,
            )
        )

    @session_app.command("switch")
    def switch_session(
        session_id: Annotated[
            str, typer.Argument(help="The ID of the session to switch to.")
        ],
    ):
        """
        Switch the active session.
        """
        # A more robust implementation would check if the session directory exists.
        _set_active_session_id(session_id)
        print(f"> Switched active session to: {session_id}")

    return app
