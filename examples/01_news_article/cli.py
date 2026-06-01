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


def create_app(
    state_type: type[StateT],
    write_article_callback: WriteArticleCallback[StateT],
) -> typer.Typer:
    """Create a CLI app that routes execution to the injected workflow callback."""
    load_dotenv()
    app = typer.Typer()

    @app.command()
    def run(
        user_input: Annotated[
            list[str] | None,
            typer.Argument(
                help="Topic for a new session (with --new), or an answer to resume an existing one."
            ),
        ] = None,
        model: Annotated[
            str,
            typer.Option(
                help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
                envvar="EXAMPLE_DEFAULT_MODEL",
            ),
        ] = "gpt-4o",
        session_id_override: Annotated[
            str | None,
            typer.Option(
                "--session-id",
                help="Explicitly specify the session ID, overriding the automatically managed one.",
            ),
        ] = None,
        new_session: Annotated[
            bool,
            typer.Option(
                "--new",
                help="Force the start of a new session. The first argument will be treated as the topic.",
            ),
        ] = False,
        stream: Annotated[
            bool, typer.Option(help="Enable streaming of LLM tokens.")
        ] = False,
    ):
        """
        Runs the news article generation workflow with automated session management.
        """
        input_text = " ".join(user_input) if user_input else None

        if new_session and not input_text:
            raise typer.BadParameter(
                "A topic is required when starting a new session with --new."
            )

        # --- Session and State Management ---
        session_dir = Path(__file__).parent / ".local"
        session_dir.mkdir(exist_ok=True)
        last_session_file = session_dir / "last_session.txt"

        session_id = session_id_override
        if session_id:
            print(f"> Using provided session ID: {session_id}")
        elif new_session:
            session_id = str(uuid.uuid4())
            print(f"> Starting new session: {session_id}")
        else:
            if last_session_file.exists():
                session_id = last_session_file.read_text().strip()
                print(f"> Resuming session {session_id}")
            else:
                if not input_text:
                    raise typer.BadParameter(
                        "No previous session found. Please start a new one with --new <topic>."
                    )
                session_id = str(uuid.uuid4())
                new_session = True
                print(
                    f"> No previous session found. Starting new session: {session_id}"
                )

        last_session_file.write_text(session_id)

        # --- Main Application Logic ---
        async def _main():
            if stream:
                print("> Streaming enabled. LLM response will appear below:")
                print("---")

            try:
                async with setup_session(
                    model=model, session_id=session_id, stream=stream
                ) as session:
                    state_store = session.get_state_store("state", state_type)
                    state = await state_store.get()

                    if new_session or state is None:
                        if input_text is None:
                            raise typer.BadParameter(
                                "A topic is required when starting a new session with --new."
                            )
                        state = state_type.model_validate(
                            {"topic": input_text, "answer": None}
                        )
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
                print('python examples/01_news_article/main.py "Your answer here"')
                print("---")

        asyncio.run(_main())

    return app
