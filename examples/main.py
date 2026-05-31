import argparse
import asyncio
import uuid
from typing import Optional

import glyff
import glyff.exceptions
import glyff_file_store
import sefia
import sefia.stores
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from pydantic import BaseModel, Field


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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, help="The topic for the news article.")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="The LLM model to use."
    )
    parser.add_argument("--session-id", type=str, help="The session ID to resume from.")
    parser.add_argument(
        "--answer", type=str, help="The user's answer to a pending question."
    )
    args = parser.parse_args()

    if not args.topic and not args.session_id:
        parser.error("Either --topic or --session-id must be provided.")

    return args


async def write_article(state: SessionState):
    researcher = Researcher(sefia.WebSearchTool())
    writer = NewsWriter(HumanInputTool(answer=state.answer))

    print("> Stage 1: Researching topic...")
    sources = await researcher.research_topic(state.topic)
    print(f"   -> Found sources: {sources}")

    print("> Stage 2: Writing article...")
    return await writer.write_article(topic=state.topic, sources=sources)


async def main():
    args = _parse_args()

    llm_client = sefia.LiteLLMClient(model=args.model)

    session_id: str = args.session_id or str(uuid.uuid4())
    serializer = PydanticSerializer()
    file_client = glyff_file_store.FileClient(
        base_dir=".local/sefia/.glyff_sessions",
        session_id=session_id,
    )
    gs = glyff.Session(
        id=session_id,
        store=glyff_file_store.JsonFileSessionStore(
            client=file_client,
            serializer=serializer,
        ),
        hasher=PydanticArgsHasher(),
    )
    sefia_store = sefia.stores.FileSessionStore(
        client=file_client,
        serializer=serializer,
    )

    if args.session_id:
        print(f"> Resuming session {session_id}")
    else:
        print("> Starting new session")

    try:
        async with gs:
            async with sefia.Session(
                llm_client=llm_client, glyff_session=gs, session_store=sefia_store
            ) as session:
                state_store = session.get_state_store("state", SessionState)
                state = await state_store.get()
                if state is None:
                    if not args.topic:
                        raise ValueError(
                            f"Session '{session_id}' not found and --topic is required to start a new one."
                        )
                    state = SessionState(
                        topic=args.topic,
                        answer=args.answer,
                    )
                else:
                    # Update topic and answer if provided via command line (for resuming with new input)
                    if args.topic:
                        state.topic = args.topic
                    if args.answer:
                        state.answer = args.answer

                await state_store.save(state)

                article = await write_article(state)

                print("\n--- FINAL ARTICLE ---")
                print(f"Title: {article.title}")
                print(f"Summary: {article.summary}")
                print(f"Sources: {', '.join(article.sources)}")
                print("---")

    except glyff.exceptions.YieldException:
        print("\n---")
        print("Session interrupted to wait for your input.")
        print("To resume, run the script again with the session ID and your answer:")
        print(
            f'python examples/sefia/main.py --session-id "{session_id}" --answer "Your answer"'
        )
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
