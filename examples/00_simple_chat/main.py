"""
00_simple_chat — the simplest possible Sefia example.

A terminal chat loop: type a message, get a reply, repeat.
Press Ctrl+C or type "exit" to quit.
"""

import asyncio
from pathlib import Path

from sefios import SessionScope

from .agent import ChatAgent

SESSION_DIR = Path(__file__).parent / ".local"

scope = SessionScope(
    session_dir=SESSION_DIR,
    stream=True,
)

agent = ChatAgent()


async def main(model: str) -> None:
    print("Simple chat — type 'exit' to quit.\n")
    session_id = "simple-chat"

    async with scope.session(session_id=session_id, model=model):
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input or user_input.lower() == "exit":
                print("Goodbye!")
                break

            response = await agent.reply(user_input)
            print(f"Assistant: {response}\n")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    model = os.getenv("EXAMPLE_DEFAULT_MODEL", "gpt-4o-mini")
    asyncio.run(main(model))
