import asyncio
import uuid
from pathlib import Path
from typing import Optional

import glyff
import glyff.exceptions
import glyff_file_store
import sefia
import sefia.stores
import sefia_litellm
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sefia.interfaces import EventHandler, Policy
from sefia.llm.events import LLMTokenReceived
from sefia.pydantic.glyff_serialization import SefiaArgsHasher, SefiaSerializer
from typing_extensions import Annotated

load_dotenv()
app = typer.Typer()


class StreamingPrintHandler(EventHandler[LLMTokenReceived]):
    """An event handler that prints LLM tokens to the console."""

    @property
    def event_types(self):
        return (LLMTokenReceived,)

    async def handle(self, event: LLMTokenReceived):
        print(event.token, end="", flush=True)


class StreamingPolicy(Policy):
    """A policy that enables console streaming of LLM tokens."""

    def create_handlers(self) -> list[EventHandler]:
        return [StreamingPrintHandler()]


class NewsArticle(BaseModel):
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]


class SessionState(BaseModel):
    """Represents the state of our long-running application."""

    topic: str
    answer: str | None = Field(default=None)


@glyff.identify("HumanInputTool")
class HumanInputTool:
    def __init__(self, answer: Optional[str] = None):
        self._answer = answer

    @sefia.tool
    async def get_human_input(self, question: str) -> str:
        """
        Asks the user a question and returns their answer.
        This tool interrupts the session to wait for user input.
        """
        if self._answer:
            return self._answer

        print(f"\n[USER_INPUT_REQUIRED] {question}\n")
        raise glyff.exceptions.YieldException()


@glyff.identify("Researcher")
class Researcher:
    def __init__(self, web_search: sefia.WebSearchTool):
        self._web = web_search

    @sefia.infer()
    async def research_topic(self, topic: str) -> list[str]:
        """
        Research the given topic to find relevant online sources.
        Your goal is to return a list of high-quality URLs related to the topic.

        **CRITICAL INSTRUCTIONS:**
        1. You MUST use the `WebSearchTool` tool to find the URLs.
        2. Do NOT answer from your own knowledge.
        3. The final answer MUST be a list of strings, where each string is a valid URL.
        """
        ...


@glyff.identify("NewsWriter")
class NewsWriter:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @sefia.infer()
    async def write_article(self, topic: str, sources: list[str]) -> NewsArticle:
        """
        Write a news article on the given topic, using the provided sources.
        1. Briefly review the sources to understand the key points.
        2. Write a draft of the article.
        3. Ask the user for feedback on the draft's direction using the HumanInputTool.
        4. Finalize the article based on the user's feedback, incorporating their suggestions.
        5. Return the final article as a NewsArticle object.
        """
        ...


async def write_article(state: SessionState):
    researcher = Researcher(sefia.WebSearchTool())
    writer = NewsWriter(HumanInputTool(answer=state.answer))

    print("> Stage 1: Researching topic...")
    sources = await researcher.research_topic(state.topic)
    print(f"\n   -> Found sources: {sources}")

    print("> Stage 2: Writing article...")
    return await writer.write_article(topic=state.topic, sources=sources)


@app.command()
def run(
    user_input: Annotated[
        list[str],
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
        llm_client = sefia_litellm.LiteLLMClient(model=model)
        serializer = SefiaSerializer()
        file_client = glyff_file_store.FileClient(
            base_dir=session_dir / "glyff_sessions",
            session_id=session_id,
        )
        gs = glyff.Session(
            id=session_id,
            store=glyff_file_store.JsonFileSessionStore(
                client=file_client, serializer=serializer
            ),
            hasher=SefiaArgsHasher(),
        )
        sefia_store = sefia.stores.FileSessionStore(
            client=file_client, serializer=serializer
        )

        policies: list[Policy] = []
        if stream:
            policies.append(StreamingPolicy())
            print("> Streaming enabled. LLM response will appear below:")
            print("---")

        try:
            async with gs:
                async with sefia.Session(
                    llm_client=llm_client,
                    glyff_session=gs,
                    session_store=sefia_store,
                    policies=policies,
                    stream=stream,
                ) as session:
                    state_store = session.get_state_store("state", SessionState)
                    state = await state_store.get()

                    if new_session or state is None:
                        state = SessionState(topic=input_text, answer=None)
                    else:
                        state.answer = input_text

                    await state_store.save(state)

                    article = await write_article(state)

                    if stream:
                        print("\n---")

                    print("\n--- FINAL ARTICLE ---")
                    print(f"Title: {article.title}")
                    print(f"Summary: {article.summary}")
                    print(f"Sources: {', '.join(article.sources)}")
                    print("---")

        except glyff.exceptions.YieldException:
            print("\n---")
            print("Session interrupted to wait for your input.")
            print("To resume, run the script again with your answer:")
            print('python examples/01_news_article/main.py "Your answer here"')
            print("---")

    asyncio.run(_main())


if __name__ == "__main__":
    app()
