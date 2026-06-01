import asyncio
import uuid
from pathlib import Path
from typing import Optional

import glyff.exceptions
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

try:
    from .main import SessionState, write_article
    from .session_setup import setup_session
except ImportError:
    from main import SessionState, write_article
    from session_setup import setup_session

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
        Optional[str],
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
            print(f"> No previous session found. Starting new session: {session_id}")

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
                state_store = session.get_state_store("state", SessionState)
                state = await state_store.get()

                if new_session or state is None:
                    if input_text is None:
                        raise typer.BadParameter(
                            "A topic is required when starting a new session with --new."
                        )
                    state = SessionState(topic=input_text, answer=None)
                else:
                    state.answer = input_text

                await state_store.save(state)

                article, sources = await write_article(state)

                if stream:
                    print("\n---")

                print("\n--- FINAL ARTICLE ---")
                print(f"Title: {article.title}")
                print(f"Summary: {article.summary}")
                print(f"Sources: {', '.join(sources)}")
                print("---")

        except glyff.exceptions.YieldException:
            print("\n---")
            print("Session interrupted to wait for your input.")
            print("To resume, run the script again with your answer:")
            print('python examples/01_news_article/runner.py "Your answer here"')
            print("---")

    asyncio.run(_main())


if __name__ == "__main__":
    app()
